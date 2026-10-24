#include "rnnt_beam.hpp"
#include "rnnt.hpp"
#include "audio_io.hpp"
#include "encoder.hpp"
#include "ggml_graph.hpp"
#include "mel.hpp"
#include "streaming_encoder.hpp"
#include "stream_clips.hpp"
#include "tokenizer.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Ids = std::vector<int32_t>;
using Rows = std::vector<pk::detail::RnntScoredTransition>;
using Beam = pk::RnntGrammarBeam;
using Policy = pk::RnntEndpointPolicy;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void fails(F&& run, const char* message) {
    try { run(); }
    catch (const std::exception& e) {
        if (std::string(e.what()).find(message) != std::string::npos) return;
        throw std::runtime_error(std::string("unexpected diagnostic: ") + e.what());
    }
    throw std::runtime_error("expected failure");
}
Ids ids(const std::vector<pk::RnntBeamToken>& tokens) {
    Ids out;
    for (const auto& token : tokens) out.push_back(token.id);
    return out;
}
std::shared_ptr<const pk::TokenGrammar> grammar(const std::string& rules,
                                               std::vector<std::string> vocab = {"a", "b", "<EOU>", "<EOB>"}) {
    return std::make_shared<const pk::TokenGrammar>(rules, vocab);
}
void advance(Beam& beam, const Rows& rows, int frames) {
    pk::detail::rnnt_advance_scored(beam, rows, frames);
}
void row(Rows& rows, int64_t frame, const Ids& prefix, int chosen, int blank = 4) {
    std::vector<float> logits((size_t)blank + 1, -20);
    logits[chosen] = 20;
    rows.push_back({frame, prefix, std::move(logits)});
}
void trace(Rows& rows, int64_t frame, Ids prefix, const Ids& suffix, int blank = 4) {
    for (int id : suffix) {
        row(rows, frame, prefix, id, blank);
        prefix.push_back(id);
    }
    row(rows, frame, prefix, blank, blank);
}
bool same_tokens(const std::vector<pk::RnntBeamToken>& a,
                 const std::vector<pk::RnntBeamToken>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].id != b[i].id || a[i].frame != b[i].frame) return false;
    return true;
}
void same(const Beam& a, const Beam& b) {
    require(a.frame() == b.frame(), "partition frame count");
    const auto& x = a.hypotheses();
    const auto& y = b.hypotheses();
    require(x.size() == y.size(), "partition beam count");
    for (size_t i = 0; i < x.size(); ++i) {
        require(x[i].score == y[i].score && x[i].grammar_state == y[i].grammar_state &&
                x[i].last_token == y[i].last_token &&
                x[i].reset_at_chunk_end == y[i].reset_at_chunk_end &&
                same_tokens(x[i].tokens, y[i].tokens) && x[i].units.size() == y[i].units.size(),
                "partition hypothesis");
        for (size_t j = 0; j < x[i].units.size(); ++j) {
            const auto& u = x[i].units[j]; const auto& v = y[i].units[j];
            require(u.endpoint_id == v.endpoint_id && u.frame == v.frame &&
                    same_tokens(u.tokens, v.tokens), "partition units");
        }
    }
}

// Enumerates tiny alignments directly against rendered word prefixes. No DFA,
// production softmax, pruning, or scorer implementation is used by this oracle.
void oracle_test(int seed, const std::vector<std::string>& vocab,
                 const std::vector<std::string>& words) {
    struct Path { Ids tokens; std::vector<int64_t> frames; long double probability; };
    const int blank = (int)vocab.size(), T = 2, cap = 2;
    auto prefix_ok = [&](const std::string& text) {
        return std::any_of(words.begin(), words.end(), [&](const auto& word) {
            return word.compare(0, text.size(), text) == 0;
        });
    };
    Rows rows;
    std::function<void(Ids, std::string)> make_rows = [&](Ids prefix, std::string text) {
        for (int t = 0; t < T; ++t) {
            int hash = seed + t * 19;
            for (int id : prefix) hash = hash * 3 + id + 1;
            std::vector<float> logits((size_t)blank + 1);
            for (int id = 0; id <= blank; ++id) logits[id] = (float)((hash + id * 7) % 17 - 8) / 5;
            rows.push_back({t, prefix, std::move(logits)});
        }
        for (int id = 0; id < blank; ++id) {
            if (!prefix_ok(text + vocab[id])) continue;
            Ids next = prefix; next.push_back(id);
            make_rows(std::move(next), text + vocab[id]);
        }
    };
    make_rows({}, "");
    std::vector<Path> paths;
    std::function<void(int, int, Ids, std::vector<int64_t>, std::string, long double)> enumerate =
        [&](int t, int emitted, Ids tokens, std::vector<int64_t> frames,
            std::string text, long double prob) {
            if (t == T) { paths.push_back({tokens, frames, prob}); return; }
            const auto found = std::find_if(rows.begin(), rows.end(), [&](const auto& r) {
                return r.frame == t && r.prefix == tokens;
            });
            require(found != rows.end(), "oracle row missing");
            long double sum = 0;
            for (float value : found->logits) sum += std::exp((long double)value);
            enumerate(t + 1, 0, tokens, frames, text,
                      prob * std::exp((long double)found->logits[blank]) / sum);
            if (emitted == cap) return;
            for (int id = 0; id < blank; ++id) {
                if (!prefix_ok(text + vocab[id])) continue;
                auto next = tokens; next.push_back(id);
                auto next_frames = frames; next_frames.push_back(t);
                enumerate(t, emitted + 1, std::move(next), std::move(next_frames), text + vocab[id],
                          prob * std::exp((long double)found->logits[id]) / sum);
            }
        };
    enumerate(0, 0, {}, {}, "", 1);
    require(paths.size() <= 64, "oracle wider than exact test beam");
    std::string rules = "root ::= ";
    for (size_t i = 0; i < words.size(); ++i) rules += (i ? " | \"" : "\"") + words[i] + '"';
    auto g = grammar(rules, vocab);
    Beam all(g, blank, 64, cap), partitioned(g, blank, 64, cap);
    advance(all, rows, T);
    advance(partitioned, rows, 1); advance(partitioned, rows, 0); advance(partitioned, rows, 1);
    same(all, partitioned);
    require(paths.size() == all.hypotheses().size(), "oracle path count");
    long double best_accepted = 0;
    for (const auto& hyp : all.hypotheses()) {
        const auto found = std::find_if(paths.begin(), paths.end(), [&](const Path& p) {
            if (p.tokens != ids(hyp.tokens) || p.frames.size() != hyp.tokens.size()) return false;
            for (size_t i = 0; i < p.frames.size(); ++i)
                if (p.frames[i] != hyp.tokens[i].frame) return false;
            return true;
        });
        require(found != paths.end(), "oracle alignment missing");
        require(std::fabs((double)std::log(found->probability) - hyp.score) < 1e-12,
                "oracle original path probability");
        std::string rendered;
        for (int id : found->tokens) rendered += vocab[id];
        const bool accepted = std::find(words.begin(), words.end(), rendered) != words.end();
        require(all.accepting(hyp) == accepted, "oracle final membership");
        if (accepted) best_accepted = std::max(best_accepted, found->probability);
        paths.erase(found);
    }
    require(all.best(true) && std::fabs(all.best(true)->score - (double)std::log(best_accepted)) < 1e-12,
            "accepted final ranking");
}

void viterbi_ranking_test() {
    const auto logits = [](float a, float b, float blank) {
        return std::vector<float>{std::log(a), std::log(b), std::log(blank)};
    };
    const Rows rows = {
        {0, {}, logits(.3f, .4f, .3f)},
        {0, {0}, logits(.05f, .05f, .9f)}, {0, {1}, logits(.05f, .05f, .9f)},
        {1, {}, logits(.65f, .1f, .25f)},
        {1, {0}, logits(.005f, .005f, .99f)}, {1, {1}, logits(.005f, .005f, .99f)}
    };
    Beam beam(grammar("root ::= \"a\" | \"b\"", {"a", "b"}), 2, 8, 1);
    advance(beam, rows, 2);
    require(beam.best(true) && ids(beam.best(true)->tokens) == Ids{1} &&
            beam.best(true)->tokens[0].frame == 0 &&
            std::fabs(std::exp(beam.best(true)->score) - .4 * .9 * .99) < 1e-7,
            "Viterbi selects best individual alignment");
    double a_sum = 0, b_sum = 0;
    for (const auto& hyp : beam.hypotheses()) {
        if (ids(hyp.tokens) == Ids{0}) a_sum += std::exp(hyp.score);
        if (ids(hyp.tokens) == Ids{1}) b_sum += std::exp(hyp.score);
    }
    require(std::fabs(a_sum - (.3 * .9 * .99 + .3 * .65 * .99)) < 1e-7 &&
            std::fabs(b_sum - (.4 * .9 * .99 + .3 * .1 * .99)) < 1e-7 && a_sum > b_sum,
            "summed-prefix winner differs; scores are not marginalized");
}

void endpoints_test() {
    auto g = grammar("root ::= \"a\" | \"ba\"");
    Rows rows;
    trace(rows, 0, {}, {0, 2, 0, 3});
    trace(rows, 1, {0, 2, 0, 3}, {1});
    Beam whole(g, 4, 1, 8, 2, 3, Policy::restart);
    Beam split(g, 4, 1, 8, 2, 3, Policy::restart);
    advance(whole, rows, 2);
    advance(split, rows, 1); advance(split, rows, 1);
    same(whole, split);
    require(whole.frame() == 2 && ids(whole.best()->tokens) == Ids{1} &&
            whole.best()->units.size() == 2 && !whole.accepting(*whole.best()),
            "early endpoints retain following incomplete suffix and all frames");
    require(whole.best()->last_token == 1 && whole.best()->cache &&
            whole.best()->reset_at_chunk_end, "no early predictor reset");
    whole.end_chunk();
    require(whole.best()->last_token == -1 && !whole.best()->predictor && !whole.best()->cache &&
            ids(whole.best()->tokens) == Ids{1}, "native-chunk reset preserves suffix");
    auto units = whole.commit_units();
    require(units.size() == 2 && units[0].endpoint_id == 2 && units[1].endpoint_id == 3 &&
            units[0].frame == 0 && units[1].frame == 0 &&
            ids(units[0].tokens) == Ids{0} && ids(units[1].tokens) == Ids{0} &&
            whole.best()->units.empty() && ids(whole.best()->tokens) == Ids{1},
            "distinct EOU/EOB commitment and retirement");
    require(whole.commit_units().empty(), "no duplicate commitment");
    trace(rows, 2, {}, {0, 2});
    advance(whole, rows, 1);
    whole.end_chunk();
    units = whole.commit_units();
    require(units.size() == 1 && ids(units[0].tokens) == Ids({1, 0}) && units[0].frame == 2 &&
            units[0].tokens[0].frame == 1 && units[0].tokens[1].frame == 2 && whole.frame() == 3,
            "suffix grammar survives reset; next-unit endpoint has absolute frames");
    whole.reset();
    require(whole.frame() == 0 && whole.best()->tokens.empty() && whole.best()->units.empty() &&
            whole.best()->last_token == -1 && !whole.best()->cache && whole.best()->score == 0,
            "fresh frontier reset");

    // Different unit text/control/frame histories cannot survive a commitment.
    rows.clear();
    rows.push_back({0, {}, {10, 9, 30, 30, -10}});
    rows.push_back({0, {0}, {-20, -20, 10, -20, -10}});
    rows.push_back({0, {1}, {-20, -20, -20, 10, -10}});
    for (int text : {0, 1}) for (int control : {2, 3}) trace(rows, 0, {text, control}, {});
    Beam alternatives(grammar("root ::= \"a\" | \"b\""), 4, 8, 2, 2, 3, Policy::restart);
    advance(alternatives, rows, 1);
    require(alternatives.best()->units.size() == 1, "winning unit exists");
    units = alternatives.commit_units();
    require(units.size() == 1 && ids(units[0].tokens) == Ids{0} && units[0].endpoint_id == 2,
            "selected chunk-final unit");
    for (const auto& h : alternatives.hypotheses())
        require(h.units.empty() && h.last_token == 2, "prune incompatible histories");

    // Compatible competing suffixes survive the same committed endpoint.
    Rows branching;
    std::function<void(Ids)> add_rows = [&](Ids prefix) {
        row(branching, 0, prefix, 4);
        if (prefix.empty()) branching.back().logits = {20, -20, -20, -20, -20};
        if (prefix == Ids{0}) branching.back().logits = {-20, -20, 20, -20, -20};
        if (prefix == Ids({0, 2})) branching.back().logits = {20, 19, -20, -20, -20};
        if (prefix.size() == 3) return;
        for (int id = 0; id < 4; ++id) { auto next = prefix; next.push_back(id); add_rows(next); }
    };
    add_rows({});
    Beam suffixes(grammar("root ::= \"a\" | \"b\""), 4, 8, 3, 2, 3, Policy::restart);
    advance(suffixes, branching, 1);
    units = suffixes.commit_units();
    require(units.size() == 1 && ids(units[0].tokens) == Ids{0}, "compatible unit selection");
    bool has_a = false, has_b = false;
    for (const auto& hyp : suffixes.hypotheses()) {
        has_a |= ids(hyp.tokens) == Ids{0}; has_b |= ids(hyp.tokens) == Ids{1};
    }
    require(has_a && has_b, "commit retains multiple compatible suffix histories");

    Rows spaced_rows; trace(spaced_rows, 0, {}, {0, 2, 0, 3, 0});
    Beam spaced(grammar("root ::= \"a\"", {"\u2581a", "b", "<EOU>", "<EOB>"}),
                4, 1, 8, 2, 3, Policy::restart);
    advance(spaced, spaced_rows, 1);
    spaced.end_chunk();
    units = spaced.commit_units();
    require(units.size() == 2 && spaced.accepting(*spaced.best()) &&
            pk::detokenize({"\u2581a", "b", "<EOU>", "<EOB>"}, ids(spaced.best()->tokens)) == "a",
            "each endpoint restarts initial-space grammar and rendering state");

    // A stream must not retain the whole sequence of already committed units.
    Beam long_stream(grammar("root ::= \"a\""), 4, 1, 4, 2, 3, Policy::restart, {1000, 10000, 4});
    for (int chunk = 0; chunk < 300; ++chunk) {
        Rows one; trace(one, chunk, {}, {0, 2});
        advance(long_stream, one, 1); long_stream.end_chunk();
        units = long_stream.commit_units();
        require(units.size() == 1 && units[0].frame == chunk &&
                long_stream.best()->tokens.empty() && long_stream.best()->units.empty() &&
                !long_stream.best()->predictor && !long_stream.best()->cache,
                "bounded retained stream history");
    }
    require(long_stream.frame() == 300, "long stream frame clock");
}

void offline_controls_test() {
    Rows rows;
    trace(rows, 0, {}, {0, 2, 1, 3, 3});
    rows[0].logits[2] = 30; // illegal before initial acceptance, even above top-k
    Beam offline(grammar("root ::= \"a\" | \"ab\""), 4, 1, 8, 2, 3);
    advance(offline, rows, 1);
    require(offline.best(true) && ids(offline.best()->tokens) == Ids({0, 1}) &&
            offline.best()->units.empty() && offline.best()->last_token == 3 &&
            !offline.best()->reset_at_chunk_end, "offline controls preserve grammar and prediction");
    offline.end_chunk();
    require(offline.best()->last_token == 3 && offline.best()->cache &&
            offline.commit_units().empty(), "offline never resets or commits endpoints");
    rows.clear(); trace(rows, 0, {}, {0, 2, 1});
    Beam partial(grammar("root ::= \"a\" | \"abc\"", {"a", "b", "<EOU>", "<EOB>", "c"}),
                 5, 1, 6, 2, 3);
    // Same trace, adjusted full model vocabulary/blank.
    rows.clear(); trace(rows, 0, {}, {0, 2, 1}, 5);
    advance(partial, rows, 1);
    require(!partial.best(true) && ids(partial.best()->tokens) == Ids({0, 1}),
            "offline marker does not waive final acceptance");
}

void boundaries_test() {
    auto a = grammar("root ::= \"a\"");
    Rows rows;
    trace(rows, 0, {}, {0}); trace(rows, 1, {0}, {});
    Beam carried(a, 4, 1, 4);
    advance(carried, rows, 1);
    const auto cache = carried.best()->cache;
    require(cache && carried.best(true), "completed ending without fabricated endpoint");
    advance(carried, rows, 1);
    require(carried.best()->cache == cache && carried.best()->tokens[0].frame == 0,
            "blank preserves prediction cache and emission frame");
    Beam unfinished(grammar("root ::= \"ab\""), 4, 1, 1);
    // At the symbol cap the scored blank advances time, not an unscored jump.
    rows.clear(); trace(rows, 0, {}, {0});
    advance(unfinished, rows, 1);
    require(!unfinished.best(true) && ids(unfinished.best()->tokens) == Ids{0}, "partial ending");
    Beam empty(a, 4);
    require(empty.best() && !empty.best(true), "empty nonnullable input");
    Beam nullable(grammar("root ::= \"\""), 4);
    require(nullable.best(true) && nullable.best()->tokens.empty(), "empty nullable input");

    std::vector<std::string> vocab(16, "<unused>"); vocab[0] = "a";
    std::vector<float> low(17, 20); low[0] = 0; low[16] = -20;
    rows = {{0, {}, low}, {0, {0}, std::vector<float>(17, 0)}};
    Beam eligible(grammar("root ::= \"a\"", vocab), 16, 1, 2);
    advance(eligible, rows, 1);
    require(eligible.best(true) && ids(eligible.best()->tokens) == Ids{0}, "legality before token top-k");
    long double sum = 0;
    for (float value : low) sum += std::exp((long double)value);
    require(std::fabs(eligible.best()->score - (double)(-std::log(sum) - std::log(17.L))) < 1e-12,
            "no legal-token renormalization");

    rows.clear();
    for (int n = 0; n <= 3; ++n)
        rows.push_back({0, Ids((size_t)n, 0), n == 3 ? std::vector<float>{0, -20, -20, -20, 1}
                                                                    : std::vector<float>{2, -20, -20, -20, 0}});
    Beam repeat(grammar("root ::= \"a\"*"), 4, 1, 3);
    advance(repeat, rows, 1);
    require(repeat.best(true) && ids(repeat.best()->tokens) == Ids({0, 0, 0}), "RNNT repeats advance grammar");
    const long double den = std::exp(2.L) + 1 + 3 * std::exp(-20.L);
    const long double cap_den = 1 + std::exp(1.L) + 3 * std::exp(-20.L);
    require(std::fabs(repeat.best()->score - (double)(6 - 3 * std::log(den) + 1 - std::log(cap_den))) < 1e-12,
            "symbol cap still pays original blank probability");
    rows.clear();
    for (int n = 0; n <= 4; ++n) rows.push_back({0, Ids((size_t)n, 2), {-20, -20, 20, -20, -20}});
    Beam controls(grammar("root ::= \"\""), 4, 1, 4, 2, 3);
    advance(controls, rows, 1);
    require(controls.best(true) && controls.best()->tokens.empty() && controls.frame() == 1,
            "repeated zero-text controls bounded by normal symbol cap");

    for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        Beam beam(a, 4, 1);
        fails([&] { advance(beam, {{0, {}, {bad, 0, 0, 0, 0}}}, 1); }, "nonfinite");
        beam.reset(); require(beam.frame() == 0, "reset after inference error");
    }
    Beam malformed(a, 4);
    fails([&] { advance(malformed, {{0, {}, {0}}}, 1); }, "shape");
    fails([&] { advance(malformed, {}, -1); }, "frame bounds");
    for (const auto limits : {pk::detail::RnntSearchLimits{0, 1000, 10},
                              pk::detail::RnntSearchLimits{100, 0, 10},
                              pk::detail::RnntSearchLimits{100, 1000, 0}}) {
        Beam limited(a, 4, 1, 4, -1, -1, Policy::preserve, limits);
        Rows one; trace(one, 0, {}, {0});
        fails([&] { advance(limited, one, 1); }, "resource limit");
    }
    fails([&] { Beam bad(nullptr, 4); }, "vocabulary");
    fails([&] { Beam bad(a, 3); }, "vocabulary");
    fails([&] { Beam bad(a, 4, 0); }, "bounds");
    fails([&] { Beam bad(a, 4, 65); }, "bounds");
    fails([&] { Beam bad(a, 4, 1, 0); }, "bounds");
    fails([&] { Beam bad(a, 4, 1, 129); }, "bounds");
    fails([&] { Beam bad(a, 4, 1, 4, 2, 2); }, "endpoint IDs");
    fails([&] { Beam bad(a, 4, 1, 4, 4); }, "endpoint IDs");
    fails([&] { Beam bad(a, 4, 1, 4, 0); }, "text token");
    auto owned = grammar("root ::= \"a\"", {"a"});
    const std::weak_ptr<const pk::TokenGrammar> weak = owned;
    Beam retained(owned, 1, 1, 1);
    owned.reset();
    rows.clear(); trace(rows, 0, {}, {0}, 1);
    advance(retained, rows, 1);
    require(!weak.expired() && retained.best(true) && ids(retained.best()->tokens) == Ids{0},
            "single-label vocabulary and retained immutable grammar lifetime");
}

std::string literal(const std::string& text) {
    std::string result = "\"";
    for (char c : text) { if (c == '\\' || c == '"') result += '\\'; result += c; }
    return result + '"';
}

int model_test() {
    const char* path = std::getenv("PARAKEET_TEST_GGUF_EOU");
    if (!path) { std::fprintf(stderr, "test_rnnt_grammar_model: PARAKEET_TEST_GGUF_EOU not set; skip\n"); return 77; }
    struct Shutdown { ~Shutdown() { pk::shutdown_backend(); } } shutdown;
    pk::ModelLoader loader;
    require(loader.load(path), "load RNNT model");
    const auto& cfg = loader.config();
    const auto& vocab = loader.tokenizer_pieces();
    int eou = -1, eob = -1;
    for (int i = 0; i < (int)vocab.size(); ++i) {
        if (vocab[i] == "<EOU>") eou = i;
        if (vocab[i] == "<EOB>") eob = i;
    }
    require(eou >= 0 && eob >= 0 && cfg.tdt_durations.empty(), "requires EOU RNNT model");
    pk::Audio audio;
    require(pk::load_audio_16k_mono("tests/fixtures/speech.wav", audio), "load fixture");
    const std::string expected = "well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait";
    auto g = grammar("root ::= " + literal(expected), vocab);
    pk::MelFrontend mel(loader);
    pk::PredictionNet pred(loader);
    pk::Joint joint(loader);
    int nm = 0, frames = 0;
    std::vector<float> features;
    mel.compute(audio.samples, features, nm, frames);
    pk::Encoder encoder(loader);
    std::vector<float> channels;
    int hidden = 0, T = 0;
    encoder.forward(features, nm, frames, channels, hidden, T);
    std::vector<float> enc(channels.size());
    for (int t = 0; t < T; ++t) for (int c = 0; c < hidden; ++c)
        enc[(size_t)t * hidden + c] = channels[(size_t)c * T + t];
    const auto legacy = pk::rnnt_greedy(pred, joint, enc, T, hidden, (int)cfg.blank_id, 10);
    auto legacy_text = legacy;
    legacy_text.erase(std::remove_if(legacy_text.begin(), legacy_text.end(),
                     [&](int id) { return id == eou || id == eob; }), legacy_text.end());
    require(pk::detokenize(vocab, legacy_text) == expected &&
            std::find(legacy.begin(), legacy.end(), eou) != legacy.end(),
            "legacy real RNNT fixture reference");
    Beam offline(g, (int)cfg.blank_id, 4, 10, eou, eob);
    const auto start = std::chrono::steady_clock::now();
    offline.advance(pred, joint, enc, T, hidden);
    std::fprintf(stderr, "RNNT grammar offline beam=4 frames=%d decode_ms=%.3f\n", T,
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    require(offline.best(true) && pk::detokenize(vocab, ids(offline.best(true)->tokens)) == expected,
            "real offline RNNT accepted reference");
    for (int width : {1, 8}) {
        Beam trial(g, (int)cfg.blank_id, width, 10, eou, eob);
        const auto began = std::chrono::steady_clock::now();
        trial.advance(pred, joint, enc, T, hidden);
        std::fprintf(stderr, "RNNT grammar offline beam=%d frames=%d decode_ms=%.3f\n", width, T,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count());
        require(trial.best(true) && pk::detokenize(vocab, ids(trial.best(true)->tokens)) == expected,
                "real RNNT practical beam widths");
    }
    Beam partitioned(g, (int)cfg.blank_id, 4, 10, eou, eob);
    for (int t = 0; t < T; t += 7) {
        const int n = std::min(7, T - t);
        const std::vector<float> chunk(enc.begin() + (size_t)t * hidden, enc.begin() + (size_t)(t + n) * hidden);
        partitioned.advance(pred, joint, chunk, n, hidden);
    }
    same(offline, partitioned);
    require(legacy == pk::rnnt_greedy(pred, joint, enc, T, hidden, (int)cfg.blank_id, 10),
            "legacy greedy unchanged after grammar decoding");
    fails([&] { offline.advance(pred, joint, {}, 1, hidden); }, "encoder shape");

    // Exercise the frontier on the real native cached encoder schedule, twice
    // in one stream. No duplicated PCM/frontend implementation is introduced.
    mel.compute(pktest::two_utterance_clip(audio.samples), features, nm, frames);
    pk::StreamingEncoder stream_encoder(loader);
    Beam stream(g, (int)cfg.blank_id, 4, 10, eou, eob, Policy::restart);
    std::vector<pk::RnntAcceptedUnit> committed;
    for (int pos = 0; pos < frames;) {
        const bool first = stream_encoder.step_num() == 0;
        const int step = first ? stream_encoder.chunk_size_first() : stream_encoder.chunk_size();
        const int hi = std::min(pos + step, frames);
        const int lo = first ? pos : std::max(0, pos - stream_encoder.pre_encode_cache_size());
        std::vector<float> window((size_t)nm * (hi - lo));
        for (int m = 0; m < nm; ++m) for (int t = lo; t < hi; ++t)
            window[(size_t)m * (hi - lo) + t - lo] = features[(size_t)m * frames + t];
        int valid = 0;
        const auto encoded = stream_encoder.step(window, hi - lo, hi == frames, valid);
        stream.advance(pred, joint, encoded, valid, cfg.d_model);
        stream.end_chunk();
        const auto units = stream.commit_units();
        for (const auto& unit : units) {
            require(pk::detokenize(vocab, ids(unit.tokens)) == expected && unit.endpoint_id == eou,
                    "real streaming accepted EOU unit");
            committed.push_back(unit);
        }
        pos = hi;
    }
    require(stream.best() && !committed.empty() && committed.front().frame > 0,
            "real multi-utterance EOU event");
    const bool complete_suffix = stream.accepting(*stream.best()) &&
        pk::detokenize(vocab, ids(stream.best()->tokens)) == expected;
    require((committed.size() == 1 && complete_suffix) ||
            (committed.size() == 2 && stream.best()->tokens.empty() &&
             committed[1].frame > committed[0].frame),
            "real stream retains both accepted utterances");
    std::fprintf(stderr, "test_rnnt_grammar_model: offline/partitioned RNNT, native cached two-utterance stream (%zu EOU), unchanged greedy passed\n", committed.size());
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--model") return model_test();
        require(argc == 1, "unexpected arguments");
        for (int seed = 0; seed < 12; ++seed) {
            oracle_test(seed, {"a", "b", "c"}, {"a", "aa", "ab"});
            oracle_test(seed, {"a", "a", "b", "ab"}, {"a", "ab"});
        }
        boundaries_test(); offline_controls_test(); endpoints_test(); viterbi_ranking_test();
        std::fprintf(stderr, "test_rnnt_grammar: independent oracle, boundaries, endpoints and retirement passed\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_rnnt_grammar: %s\n", e.what()); return 1;
    }
}
