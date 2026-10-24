#include "streaming.hpp"
#include "audio_io.hpp"
#include "ggml_graph.hpp"
#include "stream_clips.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using State = pk::detail::GrammarStreamState;
using Rows = std::vector<pk::detail::RnntScoredTransition>;
using Ids = std::vector<int32_t>;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class F> void fails(F&& run) {
    try { run(); } catch (const std::exception&) { return; }
    throw std::runtime_error("expected exception");
}
auto grammar(const std::string& rules, const std::vector<std::string>& pieces) {
    return std::make_shared<const pk::TokenGrammar>(rules, pieces);
}
void trace(Rows& rows, int64_t frame, Ids prefix, const Ids& suffix) {
    for (int token : suffix) {
        std::vector<float> logits(5, -30); logits[token] = 30;
        rows.push_back({frame, prefix, std::move(logits)});
        prefix.push_back(token);
    }
    std::vector<float> logits(5, -30); logits[4] = 30;
    rows.push_back({frame, prefix, std::move(logits)});
}
void advance(State& state, const Rows& rows, int frames) {
    pk::detail::rnnt_advance_scored(state.beam(), rows, frames);
}
void same_units(const std::vector<pk::GrammarStreamUnit>& a,
                const std::vector<pk::GrammarStreamUnit>& b) {
    require(a.size() == b.size(), "unit count differs");
    for (size_t i = 0; i < a.size(); ++i)
        require(a[i].text == b[i].text && a[i].endpoint_id == b[i].endpoint_id &&
                a[i].is_eob == b[i].is_eob && a[i].encoder_frame == b[i].encoder_frame &&
                a[i].time_sec == b[i].time_sec, "unit content/timeline differs");
}

void endpoint_test() {
    const std::vector<std::string> pieces = {"▁a", "b", "<EOU>", "<EOB>"};
    auto g = grammar("root ::= \"ab\"", pieces);
    State s(g, pieces, 4, 1, 10, 2, 3, 0.08);
    g.reset(); // frontier, not the caller, retains the immutable grammar
    Rows rows;
    trace(rows, 0, {}, {0, 1, 2, 0});
    trace(rows, 1, {0, 1, 2, 0}, {1, 3, 0});
    advance(s, rows, 2);
    require(s.beam().frame() == 2 && s.beam().best()->last_token == 0 &&
            s.beam().best()->cache && s.beam().best()->reset_at_chunk_end,
            "predictor must survive both endpoints until native chunk end");
    s.end_chunk();
    require(s.units().size() == 2 && s.units()[0].text == "ab" &&
            s.units()[1].text == "ab" && s.units()[0].endpoint_id == 2 &&
            !s.units()[0].is_eob && s.units()[1].is_eob &&
            s.units()[0].encoder_frame == 0 && s.units()[1].encoder_frame == 1 &&
            s.units()[1].time_sec == 0.08, "typed multimarker repeated units");
    require(s.partial() == "a" && s.beam().best()->last_token == -1 &&
            !s.beam().best()->cache && !s.beam().best()->predictor &&
            !s.beam().best()->reset_at_chunk_end, "suffix survives chunk-end predictor reset");
    const auto saved = s.units();
    require(!s.finalize() && s.partial() == "a", "incomplete final suffix must not commit");
    same_units(saved, s.units());
    s.acknowledge();
    require(s.units().empty() && s.partial() == "a", "ack clears only published outputs");
    // Complete the post-marker suffix on the next frame from SOS, not from
    // either marker or that suffix's old predictor. Grammar must NOT restart.
    Rows tail; trace(tail, 2, {}, {1});
    advance(s, tail, 1); s.end_chunk();
    require(s.partial() == "ab" && s.beam().frame() == 3, "all later frames processed once");
    require(s.finalize() && s.partial().empty() && s.units().size() == 1 &&
            s.units()[0].text == "ab" && s.units()[0].endpoint_id == -1 &&
            s.units()[0].encoder_frame == 3 && s.units()[0].time_sec == 0.24,
            "accepted tail without fabricated endpoint");
    s.reset();
    require(s.beam().frame() == 0 && s.units().empty() && s.partial().empty() &&
            !s.finalize(), "fresh search/output/time reset");
    advance(s, rows, 2); s.end_chunk();
    same_units(saved, s.units());

    State clean(grammar("root ::= \"ab\"", pieces), pieces, 4, 1, 10, 2, 3, 0.08);
    Rows closed; trace(closed, 0, {}, {0, 1, 2});
    advance(clean, closed, 1); clean.end_chunk(); clean.acknowledge();
    require(clean.finalize() && clean.units().empty(), "clean restart is not incomplete text");
    State nullable(grammar("root ::= \"\"", pieces), pieces, 4, 1, 10, 2, 3, 0.08);
    require(nullable.finalize() && nullable.units().empty(), "empty accepted stream");

    State same_frame(grammar("root ::= \"ab\"", pieces), pieces, 4, 1, 10, 2, 3, 0.08);
    Rows multiple; trace(multiple, 0, {}, {0, 1, 2, 0, 1, 3, 0});
    advance(same_frame, multiple, 1); same_frame.end_chunk();
    require(same_frame.units().size() == 2 && same_frame.units()[0].encoder_frame == 0 &&
            same_frame.units()[1].encoder_frame == 0 && same_frame.units()[1].is_eob &&
            same_frame.partial() == "a", "same-frame endpoints and post-marker suffix");

    State long_stream(grammar("root ::= \"ab\"", pieces), pieces, 4, 1, 10, 2, 3, 0.08);
    for (int frame = 0; frame < 300; ++frame) {
        Rows unit; trace(unit, frame, {}, {0, 1, 2});
        advance(long_stream, unit, 1); long_stream.end_chunk();
        require(long_stream.units().size() == 1 && long_stream.units()[0].text == "ab" &&
                long_stream.units()[0].encoder_frame == frame &&
                long_stream.beam().best()->units.empty() && long_stream.beam().best()->tokens.empty(),
                "repeated units must retire exactly once");
        long_stream.acknowledge();
    }
    require(long_stream.finalize() && long_stream.units().empty(), "long stream clean final suffix");
}

void acceptance_test() {
    const std::vector<std::string> pieces = {"a", "b", "<EOU>", "<EOB>"};
    State s(grammar("root ::= \"ab\"", pieces), pieces, 4, 1, 10, 2, 3, 0.1);
    // Invalid endpoints beat every text arc acoustically, but remain ineligible.
    Rows rows = {{0, {}, {-30, -30, 30, 29, 0}}};
    advance(s, rows, 1); s.end_chunk();
    require(s.units().empty() && s.partial().empty() && !s.finalize(),
            "endpoints require grammar acceptance");
    State alternatives(grammar("root ::= \"a\" | \"bb\"", pieces), pieces, 4, 2, 1, 2, 3, 0.1);
    Rows choices = {{0, {}, {1, 2, -30, -30, -20}},
                    {0, {0}, {-30, -30, -30, -30, 0}},
                    {0, {1}, {-30, -30, -30, -30, 0}}};
    advance(alternatives, choices, 1); alternatives.end_chunk();
    require(alternatives.partial() == "b", "partial reflects best unfinished path");
    require(alternatives.finalize() && alternatives.units().size() == 1 &&
            alternatives.units()[0].text == "a", "final acceptance precedes ranking");

    State revisable(grammar("root ::= \"a\" | \"bb\"", pieces), pieces, 4, 2, 1, 2, 3, 0.1);
    advance(revisable, choices, 1); revisable.end_chunk();
    require(revisable.partial() == "b" && revisable.units().empty(), "initial provisional text");
    Rows revision = {{1, {0}, {-10000, -10000, -10000, -10000, 0}},
                     {1, {1}, {30, -30, -30, -30, -30}},
                     {1, {0, 2}, {-30, -30, -30, -30, 0}},
                     {1, {0, 3}, {-30, -30, -30, -30, 0}},
                     {1, {1, 1}, {-30, -30, -30, -30, 0}}};
    advance(revisable, revision, 1); revisable.end_chunk();
    require(revisable.partial() == "a" && revisable.units().empty(), "partial must revise, not append");
}

void final_ending_rank_test() {
    const std::vector<std::string> pieces = {"a", "<EOU>"};
    for (bool acknowledge : {true, false}) for (float suffix_logit : {-2.0f, 0.0f, 2.0f}) {
        State s(grammar("root ::= \"a\"", pieces), pieces, 2, 2, 3, 1, -1, 0.08);
        Rows rows = {{0, {}, {30, -30, -30}},
                     {0, {0}, {-30, 30, -30}},
                     {0, {0, 1}, {suffix_logit, -30, 0}},
                     {0, {0, 1, 0}, {-30, -30, 30}}};
        advance(s, rows, 1); s.end_chunk();
        const bool tail_wins = suffix_logit > 0;
        const auto& paths = s.beam().hypotheses();
        require(paths.size() == 2 && paths[0].score >= paths[1].score &&
                paths[0].tokens.empty() == !tail_wins &&
                paths[1].tokens.empty() == tail_wins,
                "fixture must retain competing clean and complete endings");
        if (suffix_logit == 0)
            require(paths[0].score == paths[1].score, "equal-score endings retain existing beam order");
        require(s.units().size() == 1 && s.units()[0].text == "a" &&
                s.units()[0].endpoint_id == 1, "one shared committed EOU unit");
        if (acknowledge) s.acknowledge();
        const auto queued = s.units();
        require(s.finalize() && s.partial().empty(), "successful final ending");
        if (!tail_wins) {
            require(s.units().size() == queued.size(),
                    "higher-ranked clean ending must not duplicate the accepted unit");
            same_units(queued, s.units());
        } else {
            require(s.units().size() == queued.size() + 1,
                    "higher-ranked complete tail must beat a lower clean ending");
            same_units(queued, {s.units().begin(), s.units().end() - 1});
            const auto& tail = s.units().back();
            require(tail.text == "a" && tail.endpoint_id == -1 && !tail.is_eob &&
                    tail.encoder_frame == 1 && tail.time_sec == 0.08,
                    "real final tail must not fabricate an endpoint");
        }
    }

    const std::vector<std::string> alternatives = {"a", "b", "<EOU>"};
    // Supply every short predictor history independently of grammar/search;
    // only the explicit rows below affect the retained winning histories.
    auto table = [](int64_t frame) {
        Rows rows;
        std::function<void(Ids)> fill = [&](Ids prefix) {
            rows.push_back({frame, prefix, {-30, -30, -30, 30}});
            if (prefix.size() == 4) return;
            for (int id = 0; id < 3; ++id) {
                auto next = prefix; next.push_back(id); fill(std::move(next));
            }
        };
        fill({});
        return rows;
    };
    auto set = [](Rows& rows, const Ids& prefix, std::vector<float> logits) {
        for (auto& row : rows) if (row.prefix == prefix) { row.logits = std::move(logits); return; }
        throw std::runtime_error("missing final-ending fixture row");
    };
    {
        State initial(grammar("root ::= \"ab\"", alternatives), alternatives, 3, 2, 1, 2, -1, 0.08);
        auto rows = table(0);
        set(rows, {}, {2, -30, -30, 0});
        advance(initial, rows, 1); initial.end_chunk();
        require(initial.beam().hypotheses().size() == 2 && initial.partial() == "a" &&
                initial.beam().hypotheses()[1].tokens.empty() && initial.units().empty(),
                "incomplete prefix competes with initial empty nonnullable history");
        require(!initial.finalize() && initial.partial() == "a" && initial.units().empty(),
                "initial empty nonnullable history is not a clean successful restart");
    }
    for (float text_logit : {-2.0f, 2.0f}) {
        State nullable(grammar("root ::= \"\" | \"a\"", pieces), pieces, 2, 2, 1, 1, -1, 0.08);
        Rows rows = {{0, {}, {text_logit, -30, 0}},
                     {0, {0}, {-30, -30, 30}}, {0, {1}, {-30, -30, 30}}};
        advance(nullable, rows, 1); nullable.end_chunk();
        require(nullable.finalize() && nullable.units().size() == (text_logit > 0 ? 1u : 0u),
                "nullable empty and nonempty accepting endings use the same ranking");
        if (text_logit > 0)
            require(nullable.units()[0].text == "a" && nullable.units()[0].endpoint_id == -1,
                    "nullable nonempty winner is a tail, not an endpoint");
    }
    for (bool tail_wins : {false, true}) {
        State s(grammar("root ::= \"a\" | \"bb\"", alternatives), alternatives, 3, 3, 3, 2, -1, 0.08);
        auto rows = table(0);
        set(rows, {}, {30, -30, -30, -30});
        set(rows, {0}, {-30, -30, 30, -30});
        set(rows, {0, 2}, {tail_wins ? 0.0f : -2.0f, 2, -30, tail_wins ? -2.0f : 0.0f});
        advance(s, rows, 1); s.end_chunk();
        require(s.beam().hypotheses().size() == 3 && !s.beam().accepting(*s.beam().best()) &&
                s.partial() == "b" && s.units().size() == 1,
                "higher-scoring incomplete prefix must precede both successful endings");
        const auto queued = s.units();
        require(s.finalize() && s.partial().empty(), "skip incomplete prefix at final selection");
        require(s.units().size() == queued.size() + (tail_wins ? 1 : 0),
                "rank clean and complete endings together after skipping incomplete prefix");
        same_units(queued, {s.units().begin(), s.units().begin() + queued.size()});
        if (tail_wins) require(s.units().back().endpoint_id == -1 && s.units().back().text == "a",
                              "selected complete ending supplies final tail");
    }

    // The successful clean history can carry another uncommitted endpoint
    // while the higher-ranked incomplete history carries none. Commit only
    // the selected successful history, not the unrestricted chunk winner.
    for (bool acknowledge : {false, true}) {
        State s(grammar("root ::= \"a\" | \"bb\"", alternatives), alternatives, 3, 2, 2, 2, -1, 0.08);
        auto first = table(0);
        set(first, {}, {30, -30, -30, -30});
        set(first, {0}, {-30, -30, 30, -30});
        advance(s, first, 1); s.end_chunk();
        require(s.units().size() == 1, "initial accepted output before competing clean restart");
        if (acknowledge) s.acknowledge();
        const auto queued = s.units();
        auto second = table(1);
        set(second, {}, {0, 2, -30, -30});
        set(second, {0}, {-30, -30, 30, -30});
        advance(s, second, 1); s.end_chunk();
        require(s.partial() == "b" && s.beam().hypotheses().size() == 2 &&
                s.beam().hypotheses()[1].tokens.empty() && s.beam().hypotheses()[1].units.size() == 1,
                "lower successful history retains its own uncommitted endpoint");
        require(s.finalize() && s.partial().empty() && s.units().size() == queued.size() + 1,
                "commit selected clean history despite higher incomplete prefix");
        same_units(queued, {s.units().begin(), s.units().begin() + queued.size()});
        require(s.units().back().text == "a" && s.units().back().endpoint_id == 2 &&
                s.units().back().encoder_frame == 1 && s.units().back().time_sec == 0.08,
                "final commitment must use the same successful history and real endpoint");
    }
}

struct Window { std::vector<float> mel; int frames; bool last; };
std::vector<Window> schedule(int total, int fragment) {
    pk::detail::NativeMelChunks chunks(2, 3, 4, 2);
    std::vector<Window> out;
    auto collect = [&](const auto& mel, int frames, bool last) { out.push_back({mel, frames, last}); };
    for (int pos = 0; pos < total; pos += fragment) {
        const int n = std::min(fragment, total - pos);
        std::vector<float> block((size_t)2 * n);
        for (int m = 0; m < 2; ++m) for (int t = 0; t < n; ++t)
            block[(size_t)m * n + t] = (float)(m * 1000 + pos + t);
        chunks.append(block, n); chunks.feed(false, collect);
        require(out.empty() || !out.back().last, "mid-stream final chunk");
    }
    chunks.feed(true, collect);
    const size_t count = out.size();
    chunks.feed(true, collect);
    require(count == out.size(), "flush re-fed a native frame");
    chunks.reset(); chunks.feed(true, collect);
    require(count == out.size(), "reset retained native history");
    return out;
}
void schedule_test() {
    // Independent full-mel window oracle, including exact chunk boundaries,
    // short tails, strict final deferral, overlap and feat-major stride.
    for (int total = 0; total < 65; ++total) for (int fragment : {1, 2, 3, 4, 7, 65}) {
        const auto actual = schedule(total, fragment);
        size_t i = 0;
        for (int pos = 0; pos < total;) {
            const int size = pos == 0 ? 3 : 4;
            const int hi = std::min(total, pos + size), lo = pos == 0 ? 0 : std::max(0, pos - 2);
            require(i < actual.size(), "native chunk dropped");
            const auto& w = actual[i++];
            require(w.frames == hi - lo && w.last == (hi == total), "native size/final mismatch");
            for (int m = 0; m < 2; ++m) for (int t = lo; t < hi; ++t)
                require(w.mel[(size_t)m * w.frames + t - lo] == m * 1000 + t,
                        "native frame duplicated/dropped/overlap changed");
            pos = hi;
        }
        require(i == actual.size(), "extra native chunk");
    }
    fails([] { pk::detail::NativeMelChunks c(2, 0, 4, 2); });
    pk::detail::NativeMelChunks c(2, 3, 4, 2);
    fails([&] { c.append({1}, 1); });
}

std::string literal(const std::string& text) {
    std::string result = "\"";
    for (char c : text) { if (c == '\\' || c == '"') result += '\\'; result += c; }
    return result + '"';
}
int model_test() {
    const char* path = std::getenv("PARAKEET_TEST_GGUF_EOU");
    if (!path) return 77;
    struct Shutdown { ~Shutdown() { pk::shutdown_backend(); } } shutdown;
    pk::ModelLoader loader;
    require(loader.load(path), "load EOU RNNT model");
    pk::Audio audio;
    require(pk::load_audio_16k_mono("tests/fixtures/speech.wav", audio), "load speech.wav");
    const auto pcm = pktest::two_utterance_clip(audio.samples);
    const std::string expected = "well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait";
    auto g = grammar("root ::= " + literal(expected), loader.tokenizer_pieces());
    pk::GrammarStreamingSession stream(loader, g, 4);
    g.reset();
    std::vector<pk::GrammarStreamUnit> reference;
    int64_t reference_frames = 0;
    for (int fragment : {(int)pcm.size(), 1600, 137}) {
        stream.reset();
        std::vector<pk::GrammarStreamUnit> units;
        bool saw_partial = false, saw_live_unit = false;
        for (size_t pos = 0; pos < pcm.size(); pos += fragment) {
            stream.feed(pcm.data() + pos, (int)std::min<size_t>(fragment, pcm.size() - pos));
            saw_partial |= !stream.partial().empty();
            saw_live_unit |= !stream.units().empty();
            units.insert(units.end(), stream.units().begin(), stream.units().end());
            stream.acknowledge_output();
        }
        require(stream.finalize(), "two utterance grammar stream incomplete");
        units.insert(units.end(), stream.units().begin(), stream.units().end());
        require(units.size() == 2 && units[0].text == expected && units[1].text == expected,
                "two repeated utterances must both survive");
        require(units[0].endpoint_id >= 0 && !units[0].is_eob && !units[1].is_eob &&
                units[0].encoder_frame > 0 && units[1].encoder_frame > units[0].encoder_frame,
                "real EOU unit order/timing");
        require(saw_live_unit && (fragment == (int)pcm.size() || saw_partial),
                "incremental units/partials unavailable before finalize");
        for (const auto& unit : units)
            require(unit.time_sec == unit.encoder_frame * 0.08, "absolute event time");
        if (reference.empty()) { reference = units; reference_frames = stream.encoder_frame(); }
        else { same_units(reference, units); require(stream.encoder_frame() == reference_frames, "fragmented encoder frames"); }
        require(!stream.active() && stream.partial().empty(), "finalized lifecycle/partial");
        fails([&] { stream.feed(nullptr, 0); }); fails([&] { stream.finalize(); });
        stream.acknowledge_output();
        require(stream.units().empty(), "final ack");
    }
    // Terminal error requires a complete fresh frontend/encoder/search reset.
    stream.reset();
    fails([&] { stream.feed(nullptr, 1); });
    require(!stream.active(), "invalid feed did not fail stream");
    fails([&] { stream.feed(nullptr, 0); });
    stream.reset();
    float nan = std::numeric_limits<float>::quiet_NaN();
    fails([&] { stream.feed(&nan, 1); });
    stream.reset();
    require(stream.active() && stream.encoder_frame() == 0 && stream.units().empty(), "fresh stream reset");
    stream.feed(audio.samples.data(), (int)audio.samples.size());
    require(stream.finalize() && stream.units().size() == 1 && stream.units()[0].text == expected &&
            stream.units()[0].endpoint_id == -1,
            "reset after failure must match fresh speech and flush tail without an endpoint");
    stream.fail(); fails([&] { stream.feed(nullptr, 0); });
    stream.reset();
    require(!stream.finalize() && stream.units().empty(), "nonnullable empty stream");
    pk::GrammarStreamingSession empty(loader, grammar("root ::= \"\"", loader.tokenizer_pieces()));
    require(empty.finalize() && empty.units().empty(), "nullable empty PCM stream");
    if (const char* offline_path = std::getenv("PARAKEET_TEST_GGUF")) {
        pk::ModelLoader offline;
        require(offline.load(offline_path), "load offline hybrid model");
        fails([&] { pk::GrammarStreamingSession unsupported(offline,
            grammar("root ::= \"hello\"", offline.tokenizer_pieces())); });
    }
    std::fprintf(stderr, "real grammar PCM stream: 2 retained units, %lld native encoder frames, fragmentation/reset/error passed\n", (long long)reference_frames);
    for (const auto& unit : reference)
        std::fprintf(stderr, "  endpoint=%d frame=%lld time=%.3f text=%s\n", unit.endpoint_id,
                     (long long)unit.encoder_frame, unit.time_sec, unit.text.c_str());
    if (const char* prompt_path = std::getenv("PARAKEET_TEST_GGUF_NEMOTRON")) {
        pk::ModelLoader prompt_loader;
        require(prompt_loader.load(prompt_path) && prompt_loader.config().prompt.present,
                "load prompt-conditioned streaming model");
        pk::StreamingSession legacy(prompt_loader, "en");
        pk::run_stream_over_pcm(legacy, prompt_loader, audio.samples);
        legacy.finalize();
        require(!legacy.text().empty(), "prompt legacy reference empty");
        const auto& prompt_pieces = prompt_loader.tokenizer_pieces();
        const std::string prompt_text = "Well, I don't wish to see it any more";
        require(legacy.text().compare(0, prompt_text.size(), prompt_text) == 0,
                "prompt legacy first-clause reference");
        auto prompt_grammar = grammar("root ::= " + literal(prompt_text), prompt_pieces);
        pk::GrammarStreamingSession prompt_stream(prompt_loader, prompt_grammar, 4, "en");
        for (size_t pos = 0; pos < audio.samples.size(); pos += 1600)
            prompt_stream.feed(audio.samples.data() + pos,
                (int)std::min<size_t>(1600, audio.samples.size() - pos));
        const bool prompt_matched = prompt_stream.finalize();
        require(prompt_matched && prompt_stream.units().size() == 1 &&
                prompt_stream.units()[0].text == prompt_text && prompt_stream.units()[0].endpoint_id == -1,
                "prompt first-clause grammar streaming tail");
        fails([&] { pk::GrammarStreamingSession bad(prompt_loader, prompt_grammar, 4, "invalid-locale"); });
        std::fprintf(stderr, "prompt-conditioned grammar PCM stream: %s\n", prompt_stream.units()[0].text.c_str());
    }
    return 0;
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--model") return model_test();
        require(argc == 1, "unexpected arguments");
        final_ending_rank_test(); endpoint_test(); acceptance_test(); schedule_test();
        std::fprintf(stderr, "streaming grammar synthetic endpoints/suffixes/acceptance/native schedule passed\n");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "test_streaming_grammar: %s\n", e.what()); return 1; }
}
