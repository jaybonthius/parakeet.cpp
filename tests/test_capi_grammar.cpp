#include "parakeet_capi.h"
#include "parakeet.h"
#include "audio_io.hpp"
#include "ctc_grammar.hpp"
#include "model_loader.hpp"
#include "search.hpp"
#include "ggml_graph.hpp"
#include "tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Context = std::unique_ptr<parakeet_ctx, decltype(&parakeet_capi_free)>;
using Grammar = std::unique_ptr<parakeet_grammar, decltype(&parakeet_capi_grammar_free)>;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
double elapsed(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
std::string literal(const std::string& text) {
    std::string out = "root ::= \"";
    for (char c : text) { if (c == '"' || c == '\\') out += '\\'; out += c; }
    return out + "\"";
}
std::string ordinary(parakeet_ctx* ctx, const pk::Audio& audio, int decoder) {
    char* text = parakeet_capi_transcribe_pcm(ctx, audio.samples.data(), static_cast<int>(audio.samples.size()), 16000, decoder);
    require(text, parakeet_capi_last_error(ctx));
    std::string out(text); parakeet_capi_free_string(text); return out;
}
std::string constrained(parakeet_ctx* ctx, const parakeet_grammar* grammar, const pk::Audio& audio,
                        int decoder = 1, int beam = 4) {
    char* text = nullptr;
    int status = parakeet_capi_transcribe_pcm_grammar(ctx, grammar, audio.samples.data(), static_cast<int>(audio.samples.size()), 16000, decoder, beam, nullptr, &text);
    require(status == PARAKEET_GRAMMAR_OK, parakeet_capi_last_error(ctx));
    require(text, "OK without an owned string");
    std::string out(text); parakeet_capi_free_string(text); return out;
}

// Just enough JSON for the grammar documents: objects, arrays, strings, numbers.
struct Json {
    std::string text;
    double number = 0;
    bool integer = false;
    std::vector<Json> items;
    std::vector<std::pair<std::string, Json>> fields;
    const Json& operator[](const char* key) const {
        for (const auto& field : fields) if (field.first == key) return field.second;
        throw std::runtime_error(std::string("missing JSON key ") + key);
    }
};
struct JsonParser {
    const std::string& in;
    size_t pos = 0;
    char next() { require(pos < in.size(), "truncated JSON"); return in[pos++]; }
    void expect(char c) { require(next() == c, "malformed JSON"); }
    std::string string() {
        expect('"');
        std::string out;
        for (char c = next(); c != '"'; c = next()) {
            if (c != '\\') { out += c; continue; }
            c = next();
            require(std::strchr("\"\\/bfnrt", c) != nullptr, "unsupported JSON escape");
            out += c == 'n' ? '\n' : c == 't' ? '\t' : c == 'r' ? '\r' : c == 'b' ? '\b' : c == 'f' ? '\f' : c;
        }
        return out;
    }
    Json value() {
        Json out;
        const char c = in.at(pos);
        if (c == '"') { out.text = string(); return out; }
        if (c == '[' || c == '{') {
            ++pos;
            const char close = c == '[' ? ']' : '}';
            if (in.at(pos) == close) { ++pos; return out; }
            for (;;) {
                if (c == '[') out.items.push_back(value());
                else { auto key = string(); expect(':'); out.fields.emplace_back(std::move(key), value()); }
                const char separator = next();
                if (separator == close) return out;
                require(separator == ',', "malformed JSON container");
            }
        }
        const size_t start = pos;
        while (pos < in.size() && std::strchr("-+.eE0123456789", in[pos])) ++pos;
        const std::string digits = in.substr(start, pos - start);
        require(!digits.empty(), "unsupported JSON value");
        out.integer = digits.find_first_of(".eE") == std::string::npos;
        out.number = std::stod(digits);
        return out;
    }
};
Json parse_json(const std::string& text) {
    JsonParser parser{text};
    Json out = parser.value();
    require(parser.pos == text.size(), "trailing JSON");
    return out;
}

std::string choice(const std::vector<std::string>& phrases) {
    std::string out = "root ::= ";
    for (size_t i = 0; i < phrases.size(); ++i) out += (i ? " | " : "") + literal(phrases[i]).substr(9);
    return out;
}
const std::vector<std::string> kCommands = {
    "split view right", "next line", "kill buffer", "save file", "open file", "undo"};

// Checks one scored result (offline document or stream unit) and returns the
// per-frame difference (score - free_score) / frames.
double check_scores(const Json& result, const std::vector<std::string>& phrases, int beam,
                    const char* label) {
    const double score = result["score"].number, free_score = result["free_score"].number;
    const Json& frames = result["frames"];
    require(frames.integer && frames.number > 0, "scores need covered frames");
    require(score <= 0 && free_score <= 0, "log probabilities must not be positive");
    const auto& alternatives = result["alternatives"].items;
    require(alternatives.size() <= static_cast<size_t>(beam - 1), "more alternatives than the beam keeps");
    const auto accepted = [&](const std::string& text) {
        return std::find(phrases.begin(), phrases.end(), text) != phrases.end();
    };
    require(accepted(result["text"].text), "result text outside the grammar");
    for (size_t i = 0; i < alternatives.size(); ++i) {
        const auto& text = alternatives[i]["text"].text;
        require(accepted(text) && text != result["text"].text, "alternative not a distinct accepted text");
        for (size_t j = 0; j < i; ++j) require(alternatives[j]["text"].text != text, "duplicate alternative");
        require(i == 0 || alternatives[i - 1]["score"].number >= alternatives[i]["score"].number,
                "alternatives not ordered by score");
    }
    const double per_frame = (score - free_score) / frames.number;
    std::printf("scores %s: text=\"%s\" score=%.3f free_score=%.3f frames=%.0f per_frame=%.4f alternatives=%zu",
                label, result["text"].text.c_str(), score, free_score, frames.number, per_frame,
                alternatives.size());
    for (const auto& alternative : alternatives)
        std::printf(" [\"%s\" %.3f]", alternative["text"].text.c_str(), alternative["score"].number);
    std::printf("\n");
    return per_frame;
}

// Empty on NO_MATCH (no retained accepting path), which the plain call must agree with.
std::optional<Json> offline_scores(parakeet_ctx* ctx, const parakeet_grammar* grammar,
                                   const pk::Audio& audio, int decoder, int beam) {
    char* json = nullptr;
    const int status = parakeet_capi_transcribe_pcm_grammar_json(ctx, grammar, audio.samples.data(),
        static_cast<int>(audio.samples.size()), 16000, decoder, beam, nullptr, &json);
    char* plain = nullptr;
    const int plain_status = parakeet_capi_transcribe_pcm_grammar(ctx, grammar, audio.samples.data(),
        static_cast<int>(audio.samples.size()), 16000, decoder, beam, nullptr, &plain);
    require(status == plain_status, "scored status differs from the plain grammar call");
    if (status == PARAKEET_GRAMMAR_NO_MATCH) {
        require(!json && !plain, "NO_MATCH returned a string");
        return std::nullopt;
    }
    require(status == PARAKEET_GRAMMAR_OK && json && plain, parakeet_capi_last_error(ctx));
    Json out = parse_json(json);
    require(out["text"].text == plain, "scored text differs from the plain grammar call");
    parakeet_capi_free_string(json);
    parakeet_capi_free_string(plain);
    return out;
}

// Unrelated phrases either fail to match or score far below the free search.
void check_unrelated(const std::optional<Json>& result, int beam, const std::string& label) {
    if (!result) { std::printf("scores %s: NO_MATCH\n", label.c_str()); return; }
    require(check_scores(*result, kCommands, beam, label.c_str()) < -0.3, "unrelated phrases near free score");
}

// All units of a grammar stream over `pcm`, fed in 100 ms fragments.
std::vector<Json> stream_units(parakeet_ctx* ctx, const parakeet_grammar* grammar,
                               const std::vector<float>& pcm, int beam) {
    std::unique_ptr<parakeet_grammar_stream, decltype(&parakeet_capi_grammar_stream_free)> stream(
        parakeet_capi_grammar_stream_begin(ctx, grammar, beam, nullptr), parakeet_capi_grammar_stream_free);
    require(static_cast<bool>(stream), parakeet_capi_last_error(ctx));
    std::vector<Json> units;
    for (size_t pos = 0; pos <= pcm.size(); pos += 1600) {
        char* json = nullptr;
        const bool final = pos == pcm.size() || pos + 1600 > pcm.size();
        const int n = static_cast<int>(std::min<size_t>(1600, pcm.size() - pos));
        require(parakeet_capi_grammar_stream_feed_json(stream.get(), pcm.data() + pos, n, &json) == PARAKEET_GRAMMAR_OK && json,
                parakeet_capi_last_error(ctx));
        Json document = parse_json(json);
        parakeet_capi_free_string(json);
        for (const auto& unit : document["units"].items) units.push_back(unit);
        if (!final) continue;
        json = nullptr;
        const int status = parakeet_capi_grammar_stream_finalize_json(stream.get(), &json);
        require(status != PARAKEET_GRAMMAR_ERROR && json, parakeet_capi_last_error(ctx));
        document = parse_json(json);
        parakeet_capi_free_string(json);
        for (const auto& unit : document["units"].items) units.push_back(unit);
        break;
    }
    return units;
}

// Grammar scores: a grammar containing what was said scores close to the free
// search; unrelated phrases (or non-speech) score far below it.
void scores_test(parakeet_ctx* ctx, const pk::Audio& speech, const pk::Audio& thunder,
                 const std::string& ctc_text) {
    const int beam = 8;
    auto phrases = kCommands;
    phrases.push_back(ctc_text);
    Grammar matching(parakeet_capi_grammar_create(ctx, choice(phrases).c_str()), parakeet_capi_grammar_free);
    Grammar unrelated(parakeet_capi_grammar_create(ctx, choice(kCommands).c_str()), parakeet_capi_grammar_free);
    require(matching && unrelated, parakeet_capi_last_error(ctx));
    for (int decoder : {1, 2}) {
        const char* name = decoder == 1 ? "CTC" : "TDT";
        const auto match = offline_scores(ctx, matching.get(), speech, decoder, beam);
        require(match.has_value(), "true transcript did not match");
        const std::string label = std::string("offline ") + name + " speech/matching";
        require(check_scores(*match, phrases, beam, label.c_str()) > -0.05, "true transcript far from free score");
        if (decoder == 1)
            require((*match)["text"].text == ctc_text && (*match)["score"].number == (*match)["free_score"].number,
                    "CTC greedy transcript must score exactly the free best alignment");
        for (const auto* audio : {&speech, &thunder})
            check_unrelated(offline_scores(ctx, unrelated.get(), *audio, decoder, beam), beam,
                std::string("offline ") + name + (audio == &speech ? " speech" : " thunder") + "/unrelated");
    }
    // Path variant and output contracts.
    char* json = nullptr;
    require(parakeet_capi_transcribe_path_grammar_json(ctx, matching.get(), "tests/fixtures/speech.wav", 1, beam, nullptr, &json) == PARAKEET_GRAMMAR_OK && json,
            parakeet_capi_last_error(ctx));
    require(parse_json(json)["text"].text == ctc_text, "path variant text");
    parakeet_capi_free_string(json);
    require(parakeet_capi_transcribe_pcm_grammar_json(ctx, matching.get(), nullptr, 0, 16000, 1, beam, nullptr, nullptr) == PARAKEET_GRAMMAR_ERROR,
            "NULL output accepted");
    Grammar word(parakeet_capi_grammar_create(ctx, "root ::= \"a\""), parakeet_capi_grammar_free);
    char sentinel;
    json = &sentinel;
    require(parakeet_capi_transcribe_pcm_grammar_json(ctx, word.get(), nullptr, 0, 16000, 1, beam, nullptr, &json) == PARAKEET_GRAMMAR_NO_MATCH && !json,
            "NO_MATCH must not return JSON");
    json = &sentinel;
    require(parakeet_capi_transcribe_path_grammar_json(ctx, word.get(), "missing.wav", 1, beam, nullptr, &json) == PARAKEET_GRAMMAR_ERROR && !json &&
            *parakeet_capi_last_error(ctx), "ERROR must not return JSON");
    json = &sentinel;
    require(parakeet_capi_transcribe_pcm_grammar_json(nullptr, word.get(), nullptr, 0, 16000, 1, beam, nullptr, &json) == PARAKEET_GRAMMAR_ERROR && !json,
            "NULL ctx output contract");
}

// RNNT: offline and streaming on the cache-aware EOU model.
void rnnt_scores_test(const char* path, const pk::Audio& speech, const pk::Audio& thunder) {
    Context ctx(parakeet_capi_load(path), parakeet_capi_free);
    require(static_cast<bool>(ctx), "EOU model failed to load");
    const int beam = 8;
    const std::string spoken = "well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait";
    auto phrases = kCommands;
    phrases.push_back(spoken);
    Grammar matching(parakeet_capi_grammar_create(ctx.get(), choice(phrases).c_str()), parakeet_capi_grammar_free);
    Grammar unrelated(parakeet_capi_grammar_create(ctx.get(), choice(kCommands).c_str()), parakeet_capi_grammar_free);
    require(matching && unrelated, parakeet_capi_last_error(ctx.get()));
    const auto match = offline_scores(ctx.get(), matching.get(), speech, 2, beam);
    require(match && (*match)["text"].text == spoken, "offline RNNT lost the spoken text");
    require(check_scores(*match, phrases, beam, "offline RNNT speech/matching") > -0.05, "true transcript far from free score");
    for (const auto* audio : {&speech, &thunder})
        check_unrelated(offline_scores(ctx.get(), unrelated.get(), *audio, 2, beam), beam,
            std::string("offline RNNT ") + (audio == &speech ? "speech" : "thunder") + "/unrelated");
    // One second of trailing silence lets the model end the utterance (<EOU>).
    auto with_pause = [](const pk::Audio& audio) {
        auto pcm = audio.samples;
        pcm.insert(pcm.end(), 16000, 0.0f);
        return pcm;
    };
    const auto spoken_units = stream_units(ctx.get(), matching.get(), with_pause(speech), beam);
    require(spoken_units.size() == 1 && spoken_units[0]["text"].text == spoken &&
            spoken_units[0]["endpoint_id"].number >= 0, "stream did not end the spoken unit");
    require(check_scores(spoken_units[0], phrases, beam, "stream RNNT speech/matching") > -0.05,
            "streamed true transcript far from free score");
    // The bounded beam often keeps no accepting path for unrelated audio (no
    // unit); the widest beam is more likely to force one.
    for (const auto* audio : {&speech, &thunder})
        for (int width : {beam, 64}) {
            const auto units = stream_units(ctx.get(), unrelated.get(), with_pause(*audio), width);
            const std::string label = std::string("stream RNNT beam ") + std::to_string(width) +
                (audio == &speech ? " speech" : " thunder") + "/unrelated";
            for (const auto& unit : units) check_unrelated(unit, width, label);
            if (units.empty()) std::printf("scores %s: no accepted unit\n", label.c_str());
        }
    // A prefix that says more than the grammar's longest phrase: the forced
    // unit scores far below the free search, and the beam keeps alternatives.
    const std::vector<std::string> prefixes = {"well", "well i", "well i don't"};
    auto partial = prefixes;
    partial.insert(partial.end(), kCommands.begin(), kCommands.end());
    Grammar prefix_grammar(parakeet_capi_grammar_create(ctx.get(), choice(partial).c_str()), parakeet_capi_grammar_free);
    require(static_cast<bool>(prefix_grammar), parakeet_capi_last_error(ctx.get()));
    pk::Audio prefix;
    prefix.samples.assign(speech.samples.begin(), speech.samples.begin() + 20000); // 1.25 s
    const auto prefix_units = stream_units(ctx.get(), prefix_grammar.get(), with_pause(prefix), beam);
    require(!prefix_units.empty() && prefix_units[0]["text"].text == "well i don't", "prefix unit");
    require(check_scores(prefix_units[0], partial, beam, "stream RNNT 1.25s prefix/prefixes") < -0.3,
            "a phrase covering part of the speech must score far below the free search");
    bool alternatives = false;
    for (size_t i = 1; i < prefix_units.size(); ++i) {
        check_scores(prefix_units[i], partial, beam, "stream RNNT 1.25s prefix/prefixes (later unit)");
        alternatives |= !prefix_units[i]["alternatives"].items.empty();
    }
    require(alternatives, "stream retained no alternative");
}
}

int main() {
    try {
        char sentinel;
        char* text = &sentinel;
        require(parakeet_capi_transcribe_pcm_grammar(nullptr, nullptr, nullptr, 0, 16000, 1, 4, nullptr, &text) == PARAKEET_GRAMMAR_ERROR && !text, "NULL ctx output contract");
        require(!parakeet_capi_grammar_create(nullptr, "root ::= \"a\""), "NULL create");
        parakeet_capi_grammar_free(nullptr);
        const char* path = std::getenv("PARAKEET_TEST_GGUF");
        const char* unsupported = std::getenv("PARAKEET_TEST_GGUF_NO_CTC");
        if (!path && !unsupported) return 77;
        pk::set_num_threads(2);
        if (unsupported) {
            Context ctx(parakeet_capi_load(unsupported), parakeet_capi_free);
            require(static_cast<bool>(ctx), "unsupported model failed to load");
            Grammar grammar(parakeet_capi_grammar_create(ctx.get(), "root ::= \"a\""), parakeet_capi_grammar_free);
            require(static_cast<bool>(grammar), "RNNT vocabulary must compile without CTC");
            text = &sentinel;
            require(parakeet_capi_transcribe_pcm_grammar(ctx.get(), grammar.get(), nullptr, 0, 16000, 1, 4, nullptr, &text) == PARAKEET_GRAMMAR_ERROR && !text, "no-CTC decode error contract");
            require(std::strstr(parakeet_capi_last_error(ctx.get()), "no CTC head"), "unsupported head diagnostic");
        }
        if (!path) return 0;
        Context ctx(parakeet_capi_load(path), parakeet_capi_free);
        require(static_cast<bool>(ctx), "CTC model failed to load");
        for (const char* bad : {static_cast<const char*>(nullptr), "root ::= root", "root ::= ."}) {
            require(!parakeet_capi_grammar_create(ctx.get(), bad), "invalid rules accepted");
            require(*parakeet_capi_last_error(ctx.get()), "missing grammar error");
        }
        Grammar empty(parakeet_capi_grammar_create(ctx.get(), "root ::= \"\""), parakeet_capi_grammar_free);
        require(static_cast<bool>(empty), parakeet_capi_last_error(ctx.get()));
        text = &sentinel;
        require(parakeet_capi_transcribe_pcm_grammar(ctx.get(), empty.get(), nullptr, 0, 16000, 1, 4, nullptr, &text) == PARAKEET_GRAMMAR_OK && text && !*text, "empty acceptance contract");
        parakeet_capi_free_string(text);
        Grammar word(parakeet_capi_grammar_create(ctx.get(), "root ::= \"a\""), parakeet_capi_grammar_free);
        require(static_cast<bool>(word), parakeet_capi_last_error(ctx.get()));
        text = &sentinel;
        require(parakeet_capi_transcribe_pcm_grammar(ctx.get(), word.get(), nullptr, 0, 16000, 1, 4, nullptr, &text) == PARAKEET_GRAMMAR_NO_MATCH && !text, "no path contract");
        require(!*parakeet_capi_last_error(ctx.get()), "NO_MATCH is not an error");
        require(parakeet_capi_transcribe_pcm_grammar(ctx.get(), word.get(), nullptr, 0, 16000, 1, 4, nullptr, nullptr) == PARAKEET_GRAMMAR_ERROR, "NULL output accepted");
        float invalid = std::numeric_limits<float>::quiet_NaN();
        for (int kind = 0; kind < 5; ++kind) {
            text = &sentinel;
            int rc = parakeet_capi_transcribe_pcm_grammar(ctx.get(), kind == 0 ? nullptr : word.get(),
                kind == 4 ? &invalid : nullptr, kind == 1 ? -1 : kind == 2 || kind == 4 ? 1 : 0,
                kind == 3 ? 0 : 16000, 1, 4, nullptr, &text);
            require(rc == PARAKEET_GRAMMAR_ERROR && !text && *parakeet_capi_last_error(ctx.get()), "bad argument cleanup");
        }
        {
            Context other(parakeet_capi_load(path), parakeet_capi_free);
            require(static_cast<bool>(other), "second context load");
            text = &sentinel;
            require(parakeet_capi_transcribe_pcm_grammar(other.get(), word.get(), nullptr, 0, 16000, 1, 4, nullptr, &text) == PARAKEET_GRAMMAR_ERROR && !text, "cross-context grammar accepted");
        }
        pk::Audio audio;
        require(pk::load_audio_16k_mono("tests/fixtures/speech.wav", audio), "fixture load");
        const std::string default_before = ordinary(ctx.get(), audio, 0);
        const std::string ctc = ordinary(ctx.get(), audio, 1);
        std::string rules = literal(ctc);
        const auto compile_start = Clock::now();
        Grammar grammar(parakeet_capi_grammar_create(ctx.get(), rules.c_str()), parakeet_capi_grammar_free);
        const double compile_ms = elapsed(compile_start);
        require(static_cast<bool>(grammar), parakeet_capi_last_error(ctx.get()));
        rules.assign(rules.size(), 'x'); // create must retain no caller buffer
        require(constrained(ctx.get(), grammar.get(), audio) == ctc, "accepted unconstrained optimum changed");
        require(ordinary(ctx.get(), audio, 0) == default_before, "default decode changed");
        Grammar forced(parakeet_capi_grammar_create(ctx.get(), "root ::= \"a\""), parakeet_capi_grammar_free);
        require(constrained(ctx.get(), forced.get(), audio) == "a", "real audio not constrained during search");
        // This deliberately unrelated forced output demonstrates why OK is not acoustic rejection.
        require(parakeet_capi_set_concurrency(ctx.get(), 2, 2) == 2, "pool setup");
        auto first = std::async(std::launch::async, [&] { return constrained(ctx.get(), grammar.get(), audio); });
        auto second = std::async(std::launch::async, [&] { return constrained(ctx.get(), grammar.get(), audio); });
        require(first.get() == ctc && second.get() == ctc, "shared immutable grammar concurrency");
        parakeet_capi_set_concurrency(ctx.get(), 1, 2);
        double greedy_ms = 0, grammar_ms = 0;
        for (int i = 0; i < 3; ++i) {
            auto start = Clock::now();
            require(ordinary(ctx.get(), audio, 1) == ctc, "ordinary CTC regression");
            greedy_ms += elapsed(start);
            start = Clock::now();
            require(constrained(ctx.get(), grammar.get(), audio) == ctc, "grammar regression");
            grammar_ms += elapsed(start);
        }
        float* matrix = nullptr; int frames = 0, classes = 0;
        require(parakeet_capi_transcribe_pcm_logits(ctx.get(), audio.samples.data(), static_cast<int>(audio.samples.size()), 16000, &matrix, &frames, &classes) == 0, "logits acquisition");
        std::vector<float> probs(matrix, matrix + static_cast<size_t>(frames) * classes);
        parakeet_capi_free_logits(matrix);
        // Metadata is read from the same immutable GGUF solely for decode-only timing.
        pk::ModelLoader loader;
        require(loader.load(path), "metadata load");
        auto start = Clock::now();
        pk::CtcGrammar compiled(literal(ctc), loader.tokenizer_pieces(), static_cast<int>(loader.config().blank_id));
        double core_compile_ms = elapsed(start);
        start = Clock::now();
        for (int i = 0; i < 50; ++i) require(compiled.decode(probs, frames, classes).matched, "cached decode");
        double search_ms = elapsed(start) / 50;
        start = Clock::now();
        for (int i = 0; i < 50; ++i) pk::ctc_greedy(probs, frames, classes, static_cast<int>(loader.config().blank_id));
        double greedy_search_ms = elapsed(start) / 50;
        pk::Audio thunder;
        require(pk::load_audio_16k_mono("third_party/ced.cpp/benchmarks/demo/clips/thunder.wav", thunder),
                "thunder fixture load");
        scores_test(ctx.get(), audio, thunder, ctc);
        if (const char* eou = std::getenv("PARAKEET_TEST_GGUF_EOU")) rnnt_scores_test(eou, audio, thunder);
        else std::printf("scores: RNNT skipped, PARAKEET_TEST_GGUF_EOU unset\n");
        std::printf("test_capi_grammar: PASS\nCTC: %s\n", ctc.c_str());
        std::printf("threads=2 compile_ms=%.3f core_compile_ms=%.3f frames=%d classes=%d greedy_search_ms=%.3f grammar_search_ms=%.3f greedy_e2e_ms=%.3f grammar_e2e_ms=%.3f (3 interleaved pairs)\n",
                    compile_ms, core_compile_ms, frames, classes, greedy_search_ms, search_ms, greedy_ms / 3, grammar_ms / 3);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_capi_grammar: %s\n", e.what());
        return 1;
    }
}
