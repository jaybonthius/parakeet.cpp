#include "parakeet_capi.h"
#include "parakeet.h"
#include "audio_io.hpp"
#include "ctc_grammar.hpp"
#include "model_loader.hpp"
#include "search.hpp"
#include "ggml_graph.hpp"
#include "tokenizer.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
std::string constrained(parakeet_ctx* ctx, const parakeet_grammar* grammar, const pk::Audio& audio) {
    char* text = nullptr;
    int status = parakeet_capi_transcribe_pcm_grammar(ctx, grammar, audio.samples.data(), static_cast<int>(audio.samples.size()), 16000, 1, 4, nullptr, &text);
    require(status == PARAKEET_GRAMMAR_OK, parakeet_capi_last_error(ctx));
    require(text, "OK without an owned string");
    std::string out(text); parakeet_capi_free_string(text); return out;
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
        std::printf("test_capi_grammar: PASS\nCTC: %s\n", ctc.c_str());
        std::printf("threads=2 compile_ms=%.3f core_compile_ms=%.3f frames=%d classes=%d greedy_search_ms=%.3f grammar_search_ms=%.3f greedy_e2e_ms=%.3f grammar_e2e_ms=%.3f (3 interleaved pairs)\n",
                    compile_ms, core_compile_ms, frames, classes, greedy_search_ms, search_ms, greedy_ms / 3, grammar_ms / 3);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_capi_grammar: %s\n", e.what());
        return 1;
    }
}
