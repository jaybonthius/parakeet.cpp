#include "parakeet_capi.h"
#include "parakeet.h"
#include "audio_io.hpp"
#include "ggml_graph.hpp"
#include "stream_clips.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Context = std::unique_ptr<parakeet_ctx, decltype(&parakeet_capi_free)>;
using Grammar = std::unique_ptr<parakeet_grammar, decltype(&parakeet_capi_grammar_free)>;
using Stream = std::unique_ptr<parakeet_grammar_stream, decltype(&parakeet_capi_grammar_stream_free)>;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
std::string literal(const std::string& text) {
    std::string out = "root ::= \"";
    for (char c : text) { if (c == '"' || c == '\\') out += '\\'; out += c; }
    return out + '"';
}
std::string array(const std::string& json, const char* key, const char* next) {
    const std::string start = std::string("\"") + key + "\":[";
    size_t pos = json.find(start);
    require(pos != std::string::npos, "missing JSON array");
    pos += start.size();
    size_t end = next ? json.find(std::string("],\"") + next + "\":", pos) : json.rfind("]}");
    require(end != std::string::npos && end >= pos, "malformed JSON arrays");
    return json.substr(pos, end - pos);
}
struct Output {
    std::string units, events;
    bool partial = false, live_unit = false;
    double first_partial_ms = -1, first_unit_ms = -1, compute_ms = 0;
};
void collect(Output& result, const std::string& json, bool live, double elapsed) {
    require(json.find("{\"partial\":\"") == 0, "missing partial");
    auto units = array(json, "units", "events"), events = array(json, "events", nullptr);
    bool partial = json.find("{\"partial\":\"\"") != 0;
    if (partial && result.first_partial_ms < 0) result.first_partial_ms = elapsed;
    if (!units.empty() && result.first_unit_ms < 0) result.first_unit_ms = elapsed;
    result.partial |= partial;
    result.live_unit |= live && !units.empty();
    if (!units.empty()) { if (!result.units.empty()) result.units += ','; result.units += units; }
    if (!events.empty()) { if (!result.events.empty()) result.events += ','; result.events += events; }
}
Output run(parakeet_ctx* ctx, parakeet_grammar_stream* stream, const std::vector<float>& pcm,
           int fragment, int final_status = PARAKEET_GRAMMAR_OK) {
    Output result;
    auto start = Clock::now();
    auto elapsed = [&] { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); };
    for (size_t pos = 0; pos < pcm.size(); pos += fragment) {
        char* json = nullptr;
        require(parakeet_capi_grammar_stream_feed_json(stream, pcm.data() + pos,
            static_cast<int>(std::min<size_t>(fragment, pcm.size() - pos)), &json) == PARAKEET_GRAMMAR_OK && json,
            parakeet_capi_last_error(ctx));
        collect(result, json, true, elapsed());
        parakeet_capi_free_string(json);
    }
    char* json = nullptr;
    require(parakeet_capi_grammar_stream_finalize_json(stream, &json) == final_status && json,
            "final status/JSON ownership");
    collect(result, json, false, elapsed());
    require(final_status != PARAKEET_GRAMMAR_OK || std::string(json).find("{\"partial\":\"\"") == 0,
            "complete finalize must clear partial");
    parakeet_capi_free_string(json);
    result.compute_ms = elapsed();
    return result;
}
void terminal(parakeet_grammar_stream* stream) {
    char sentinel;
    char* out = &sentinel;
    require(parakeet_capi_grammar_stream_feed_json(stream, nullptr, 0, &out) == PARAKEET_GRAMMAR_ERROR && !out,
            "terminal feed must refuse");
    out = &sentinel;
    require(parakeet_capi_grammar_stream_finalize_json(stream, &out) == PARAKEET_GRAMMAR_ERROR && !out,
            "terminal finalize must refuse");
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 1 || (argc == 2 && std::string(argv[1]) == "prompt"), "bad mode");
        const bool prompt = argc == 2;
        char sentinel;
        char* json = &sentinel;
        require(parakeet_capi_grammar_stream_feed_json(nullptr, nullptr, 0, &json) == PARAKEET_GRAMMAR_ERROR && !json,
                "NULL stream feed");
        json = &sentinel;
        require(parakeet_capi_grammar_stream_finalize_json(nullptr, &json) == PARAKEET_GRAMMAR_ERROR && !json,
                "NULL stream finalize");
        require(parakeet_capi_grammar_stream_reset(nullptr) == PARAKEET_GRAMMAR_ERROR, "NULL reset");
        require(!parakeet_capi_grammar_stream_begin(nullptr, nullptr, 4, nullptr), "NULL begin");
        parakeet_capi_grammar_stream_free(nullptr);
        const char* path = std::getenv(prompt ? "PARAKEET_TEST_GGUF_NEMOTRON" : "PARAKEET_TEST_GGUF_EOU");
        if (!path) return 77;
        pk::set_num_threads(2);
        Context ctx(parakeet_capi_load(path), parakeet_capi_free);
        require(static_cast<bool>(ctx), "load streaming model");
        pk::Audio audio;
        require(pk::load_audio_16k_mono("tests/fixtures/speech.wav", audio), "load speech");
        const std::string expected = prompt ? "Well, I don't wish to see it any more" :
            "well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait";
        Grammar grammar(parakeet_capi_grammar_create(ctx.get(), literal(expected).c_str()), parakeet_capi_grammar_free);
        require(static_cast<bool>(grammar), parakeet_capi_last_error(ctx.get()));
        for (int beam : {0, 65}) require(!parakeet_capi_grammar_stream_begin(ctx.get(), grammar.get(), beam, "en"), "bad beam accepted");
        Stream stream(parakeet_capi_grammar_stream_begin(ctx.get(), grammar.get(), 4, "en"), parakeet_capi_grammar_stream_free);
        require(static_cast<bool>(stream), parakeet_capi_last_error(ctx.get()));
        Stream concurrent(parakeet_capi_grammar_stream_begin(ctx.get(), grammar.get(), 4, "en"), parakeet_capi_grammar_stream_free);
        require(static_cast<bool>(concurrent), "second stream");
        grammar.reset(); // streams must retain the compiled DFA, not this handle
        const auto pcm = prompt ? audio.samples : pktest::two_utterance_clip(audio.samples);
        Output reference;
        for (int fragment : {static_cast<int>(pcm.size()), 1600, 137}) {
            require(parakeet_capi_grammar_stream_reset(stream.get()) == PARAKEET_GRAMMAR_OK, "fresh reset");
            auto output = run(ctx.get(), stream.get(), pcm, fragment);
            require(output.units.find(expected) != std::string::npos, "lost accepted text");
            if (prompt) {
                require(output.units.find("\"endpoint_id\":-1") != std::string::npos && output.events.empty(), "prompt tail fabricated event");
            } else {
                require(output.units.find(expected) != output.units.rfind(expected), "second utterance lost");
                require(output.events == "{\"type\":\"eou\",\"id\":1024,\"frame\":98,\"t\":7.840000},{\"type\":\"eou\",\"id\":1024,\"frame\":194,\"t\":15.520000}",
                        "real EOU frames/type/timeline");
                require(output.live_unit && (fragment == static_cast<int>(pcm.size()) || output.partial), "not genuinely incremental");
            }
            if (reference.units.empty()) reference = output;
            else require(reference.units == output.units && reference.events == output.events, "external PCM partition changed accepted units/events");
            std::printf("%s stream threads=2 beam=4 fragment=%d compute_ms=%.3f first_partial_compute_ms=%.3f first_unit_compute_ms=%.3f\n",
                prompt ? "prompt" : "EOU", fragment, output.compute_ms, output.first_partial_ms, output.first_unit_ms);
            terminal(stream.get());
        }
        require(parakeet_capi_set_concurrency(ctx.get(), 2, 2) == 2, "stream pool");
        require(parakeet_capi_grammar_stream_reset(stream.get()) == PARAKEET_GRAMMAR_OK, "pooled reset");
        auto first = std::async(std::launch::async, [&] { return run(ctx.get(), stream.get(), pcm, 1600); });
        auto second = std::async(std::launch::async, [&] { return run(ctx.get(), concurrent.get(), pcm, 137); });
        require(first.get().units == reference.units && second.get().units == reference.units, "concurrent shared grammar streams");
        parakeet_capi_set_concurrency(ctx.get(), 1, 2);
        if (prompt) return 0;
        for (int bad : {0, 1, 2, 3}) {
            require(parakeet_capi_grammar_stream_reset(stream.get()) == PARAKEET_GRAMMAR_OK, "reset before error");
            float nan = std::numeric_limits<float>::quiet_NaN();
            json = &sentinel;
            require(parakeet_capi_grammar_stream_feed_json(stream.get(), bad == 2 ? &nan : nullptr,
                bad == 0 ? -1 : bad == 1 || bad == 2 ? 1 : 0, bad == 3 ? nullptr : &json) == PARAKEET_GRAMMAR_ERROR,
                "bad feed accepted");
            require(bad == 3 || !json, "error JSON not NULL");
            terminal(stream.get());
        }
        require(parakeet_capi_grammar_stream_reset(stream.get()) == PARAKEET_GRAMMAR_OK, "reset after error");
        auto tail = run(ctx.get(), stream.get(), audio.samples, 1600);
        require(tail.units.find("\"endpoint_id\":-1") != std::string::npos && tail.events.empty(), "tail fabricated endpoint");
        require(tail.units.find(expected) != std::string::npos, "reset lost fresh utterance");
        require(parakeet_capi_grammar_stream_reset(stream.get()) == PARAKEET_GRAMMAR_OK, "empty reset");
        json = nullptr;
        require(parakeet_capi_grammar_stream_finalize_json(stream.get(), &json) == PARAKEET_GRAMMAR_NO_MATCH && json &&
            std::string(json) == "{\"partial\":\"\",\"units\":[],\"events\":[]}", "empty nonnullable NO_MATCH JSON");
        parakeet_capi_free_string(json);
        terminal(stream.get());
        require(parakeet_capi_grammar_stream_reset(stream.get()) == PARAKEET_GRAMMAR_OK, "incomplete remainder reset");
        auto incomplete = audio.samples;
        incomplete.insert(incomplete.end(), 9600, 0.0f);
        incomplete.insert(incomplete.end(), audio.samples.begin(), audio.samples.begin() + 48000);
        auto remainder = run(ctx.get(), stream.get(), incomplete, 1600, PARAKEET_GRAMMAR_NO_MATCH);
        require(remainder.units.find(expected) != std::string::npos && remainder.events.find("eou") != std::string::npos && remainder.partial,
                "NO_MATCH retracted prior accepted unit or lost partial suffix");
        Grammar empty(parakeet_capi_grammar_create(ctx.get(), "root ::= \"\""), parakeet_capi_grammar_free);
        Stream nullable(parakeet_capi_grammar_stream_begin(ctx.get(), empty.get(), 4, nullptr), parakeet_capi_grammar_stream_free);
        require(static_cast<bool>(nullable), "nullable stream");
        auto accepted_empty = run(ctx.get(), nullable.get(), {}, 1);
        require(accepted_empty.units.empty() && accepted_empty.events.empty(), "empty stream invented a unit");
        std::puts("C API grammar stream PASS: lifetime, fragmentation, concurrency, reset/error/finalize and incomplete remainder");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_capi_grammar_stream: %s\n", e.what());
        return 1;
    }
}
