#include "parakeet_capi.h"
#include "parakeet.h"
#include "audio_io.hpp"
#include "ggml_graph.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Context = std::unique_ptr<parakeet_ctx, decltype(&parakeet_capi_free)>;
using Grammar = std::unique_ptr<parakeet_grammar, decltype(&parakeet_capi_grammar_free)>;
using Clock = std::chrono::steady_clock;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
std::string literal(const std::string& text) {
    std::string out = "root ::= \"";
    for (char c : text) { if (c == '"' || c == '\\') out += '\\'; out += c; }
    return out + '"';
}
std::string decode(parakeet_ctx* ctx, const parakeet_grammar* grammar,
                   const pk::Audio& audio, int decoder, bool path = false) {
    char* text = nullptr;
    int status = path
        ? parakeet_capi_transcribe_path_grammar(ctx, grammar, "tests/fixtures/speech.wav", decoder, 4, "en", &text)
        : parakeet_capi_transcribe_pcm_grammar(ctx, grammar, audio.samples.data(),
            static_cast<int>(audio.samples.size()), 16000, decoder, 4, "en", &text);
    require(status == PARAKEET_GRAMMAR_OK && text, parakeet_capi_last_error(ctx));
    std::string result(text);
    parakeet_capi_free_string(text);
    return result;
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected hybrid, rnnt or prompt");
        const std::string family = argv[1];
        const bool hybrid = family == "hybrid", prompt = family == "prompt";
        require(hybrid || prompt || family == "rnnt", "unknown family");
        const char* path = std::getenv(hybrid ? "PARAKEET_TEST_GGUF" : prompt ?
            "PARAKEET_TEST_GGUF_NEMOTRON" : "PARAKEET_TEST_GGUF_EOU");
        if (!path) return 77;
        pk::set_num_threads(2);
        Context ctx(parakeet_capi_load(path), parakeet_capi_free);
        require(static_cast<bool>(ctx), "load ASR context");
        pk::Audio audio;
        require(pk::load_audio_16k_mono("tests/fixtures/speech.wav", audio), "load speech");
        const std::string expected = hybrid
            ? "Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait."
            : prompt ? "Well, I don't wish to see it any more"
            : "well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait";
        const auto start = Clock::now();
        Grammar grammar(parakeet_capi_grammar_create(ctx.get(), literal(expected).c_str()), parakeet_capi_grammar_free);
        const double compile_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        require(static_cast<bool>(grammar), parakeet_capi_last_error(ctx.get()));
        for (int decoder : {0, 2}) {
            require(decode(ctx.get(), grammar.get(), audio, decoder) == expected, "PCM transducer grammar");
            require(decode(ctx.get(), grammar.get(), audio, decoder, true) == expected, "path transducer grammar");
        }
        char sentinel;
        char* text = &sentinel;
        std::vector<float> pcm32k;
        for (float sample : audio.samples) { pcm32k.push_back(sample); pcm32k.push_back(sample); }
        for (int decoder : hybrid ? std::vector<int>{1, 2} : std::vector<int>{2}) {
            require(parakeet_capi_transcribe_pcm_grammar(ctx.get(), grammar.get(), pcm32k.data(),
                static_cast<int>(pcm32k.size()), 32000, decoder, 4, "en", &text) == PARAKEET_GRAMMAR_OK &&
                text && text == expected, "PCM resampling changed constrained output");
            parakeet_capi_free_string(text);
        }
        if (hybrid) {
            Grammar alternative(parakeet_capi_grammar_create(ctx.get(), "root ::= \"well\" | \"Well\""), parakeet_capi_grammar_free);
            require(static_cast<bool>(alternative), "compile alternatives");
            require(decode(ctx.get(), alternative.get(), audio, 0) == "Well" &&
                    decode(ctx.get(), alternative.get(), audio, 2) == "Well" &&
                    decode(ctx.get(), alternative.get(), audio, 1) == "well",
                    "hybrid default must actually select TDT, not CTC");
            require(!parakeet_capi_grammar_stream_begin(ctx.get(), grammar.get(), 4, nullptr), "offline hybrid must not fake streaming");
        } else {
            require(parakeet_capi_transcribe_pcm_grammar(ctx.get(), grammar.get(), nullptr, 0, 16000,
                1, 4, "en", &text) == PARAKEET_GRAMMAR_ERROR && !text, "missing selected CTC head");
            require(std::strstr(parakeet_capi_last_error(ctx.get()), "no CTC head"), "head diagnostic");
        }
        for (int decoder : {-1, 3}) {
            text = &sentinel;
            require(parakeet_capi_transcribe_pcm_grammar(ctx.get(), grammar.get(), nullptr, 0,
                16000, decoder, 4, "en", &text) == PARAKEET_GRAMMAR_ERROR && !text, "invalid decoder");
        }
        for (int beam : {0, 65}) {
            text = &sentinel;
            require(parakeet_capi_transcribe_pcm_grammar(ctx.get(), grammar.get(), nullptr, 0,
                16000, 0, beam, "en", &text) == PARAKEET_GRAMMAR_ERROR && !text, "invalid beam");
        }
        text = &sentinel;
        require(parakeet_capi_transcribe_pcm_grammar(ctx.get(), grammar.get(), nullptr, 0,
            16000, 0, 4, "en", &text) == PARAKEET_GRAMMAR_NO_MATCH && !text &&
            !*parakeet_capi_last_error(ctx.get()), "empty nonnullable transducer request");
        Grammar empty(parakeet_capi_grammar_create(ctx.get(), "root ::= \"\""), parakeet_capi_grammar_free);
        require(static_cast<bool>(empty), "empty grammar compile");
        require(parakeet_capi_transcribe_pcm_grammar(ctx.get(), empty.get(), nullptr, 0,
            16000, 2, 4, "en", &text) == PARAKEET_GRAMMAR_OK && text && !*text, "empty transducer acceptance");
        parakeet_capi_free_string(text);
        float tiny = 0;
        require(parakeet_capi_transcribe_pcm_grammar(ctx.get(), empty.get(), &tiny, 1,
            32000, 2, 4, "en", &text) == PARAKEET_GRAMMAR_OK && text && !*text, "resampled empty PCM acceptance");
        parakeet_capi_free_string(text);
        for (const char* bad_path : {static_cast<const char*>(nullptr), "missing-grammar-audio.wav"}) {
            text = &sentinel;
            require(parakeet_capi_transcribe_path_grammar(ctx.get(), grammar.get(), bad_path, 0,
                4, "en", &text) == PARAKEET_GRAMMAR_ERROR && !text, "path error ownership");
        }
        if (prompt) {
            text = &sentinel;
            require(parakeet_capi_transcribe_pcm_grammar(ctx.get(), empty.get(), nullptr, 0,
                16000, 0, 4, "invalid-locale", &text) == PARAKEET_GRAMMAR_ERROR && !text, "unknown prompt locale");
            require(!parakeet_capi_grammar_stream_begin(ctx.get(), grammar.get(), 4, "invalid-locale"), "unknown stream locale");
        }
        Context other(parakeet_capi_load(path), parakeet_capi_free);
        require(static_cast<bool>(other), "second context");
        text = &sentinel;
        require(parakeet_capi_transcribe_path_grammar(other.get(), grammar.get(), "tests/fixtures/speech.wav",
            0, 4, "en", &text) == PARAKEET_GRAMMAR_ERROR && !text, "cross-context path");
        require(!parakeet_capi_grammar_stream_begin(other.get(), grammar.get(), 4, "en"), "cross-context stream");
        other.reset();
        require(parakeet_capi_set_concurrency(ctx.get(), 2, 2) == 2, "pool setup");
        auto first = std::async(std::launch::async, [&] { return decode(ctx.get(), grammar.get(), audio, 0); });
        auto second = std::async(std::launch::async, [&] { return decode(ctx.get(), grammar.get(), audio, 2); });
        require(first.get() == expected && second.get() == expected, "concurrent transducer grammar sharing");
        parakeet_capi_set_concurrency(ctx.get(), 1, 2);
        double latency = 0;
        for (int i = 0; i < 3; ++i) {
            auto begin = Clock::now();
            require(decode(ctx.get(), grammar.get(), audio, 0) == expected, "repeated decode");
            latency += std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        }
        std::printf("%s C API grammar dispatch PASS: threads=2 beam=4 compile_ms=%.3f e2e_ms=%.3f (3 warm runs)\n",
                    family.c_str(), compile_ms, latency / 3);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_capi_grammar_dispatch: %s\n", e.what());
        return 1;
    }
}
