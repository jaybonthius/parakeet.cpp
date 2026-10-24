#include "tdt.hpp"
#include "audio_io.hpp"
#include "encoder.hpp"
#include "ggml_graph.hpp"
#include "mel.hpp"
#include "tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using pk::detail::TdtScoredTransition;
using Rows = std::vector<TdtScoredTransition>;
using Ids = std::vector<int32_t>;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template<class F>
void fails(F&& run, const char* message) {
    try { run(); }
    catch (const std::exception& e) {
        require(std::string(e.what()).find(message) != std::string::npos,
                "unexpected error diagnostic");
        return;
    }
    throw std::runtime_error("expected failure");
}

Ids ids(const pk::TdtBeamHypothesis& hyp) {
    Ids result;
    for (const auto& token : hyp.tokens) result.push_back(token.id);
    return result;
}

std::vector<std::string> pieces(int count = 16) {
    std::vector<std::string> result{"a", "b", "c"};
    for (int i = 3; i < count; ++i)
        result.push_back("<unused" + std::to_string(i) + ">");
    return result;
}

std::string literal(const std::string& text) {
    std::string out = "\"";
    for (char c : text) {
        if (c == '\\' || c == '"') out += '\\';
        out += c;
    }
    return out + '"';
}

std::string rules(const std::vector<std::string>& words) {
    std::string out = "root ::= ";
    for (size_t i = 0; i < words.size(); ++i) {
        if (i) out += " | ";
        out += literal(words[i]);
    }
    return out;
}

const std::vector<float>& lookup(const Rows& rows, int frame, const Ids& prefix) {
    for (const auto& row : rows)
        if (row.frame == frame && row.prefix == prefix) return row.logits;
    throw std::runtime_error("oracle missing logits");
}

// Exhaustive alignment enumeration uses rendered finite-language membership,
// not TokenGrammar transitions, beam pruning, or the production softmax.
struct Oracle {
    const Rows& rows;
    const std::vector<std::string>& vocab;
    const std::vector<std::string>& words;
    const std::vector<int32_t>& durations;
    int T;
    std::map<std::pair<Ids, int64_t>, long double> probability;

    bool prefix(const std::string& text) const {
        return std::any_of(words.begin(), words.end(), [&](const std::string& word) {
            return word.compare(0, text.size(), text) == 0;
        });
    }

    void enumerate(int64_t frame, Ids history, const std::string& text,
                   long double prob = 1) {
        if (frame >= T) {
            if (std::find(words.begin(), words.end(), text) != words.end())
                probability[{history, frame}] += prob;
            return;
        }
        const auto& logits = lookup(rows, (int)frame, history);
        const size_t blank = vocab.size();
        long double token_sum = 0, duration_sum = 0;
        for (size_t i = 0; i <= blank; ++i) token_sum += std::exp((long double)logits[i]);
        for (size_t i = 0; i < durations.size(); ++i)
            duration_sum += std::exp((long double)logits[blank + 1 + i]);
        for (size_t d = 0; d < durations.size(); ++d) {
            const long double dp = std::exp((long double)logits[blank + 1 + d]) / duration_sum;
            for (size_t token = 0; token <= blank; ++token) {
                const long double tp = std::exp((long double)logits[token]) / token_sum;
                if (token == blank) {
                    if (durations[d] > 0)
                        enumerate(frame + durations[d], history, text, prob * tp * dp);
                } else if (prefix(text + vocab[token])) {
                    Ids next = history;
                    next.push_back((int32_t)token);
                    enumerate(frame + durations[d], std::move(next), text + vocab[token], prob * tp * dp);
                }
            }
        }
    }
};

void finite_oracle(const std::vector<std::string>& vocab,
                   const std::vector<std::string>& words,
                   const std::vector<int32_t>& durations, int seed) {
    const int T = 2;
    Rows rows;
    std::vector<Ids> histories{{}};
    for (size_t i = 0; i < histories.size(); ++i) {
        const Ids history = histories[i];
        std::string text;
        for (int32_t token : history) text += vocab[token];
        for (int frame = 0; frame < T; ++frame) {
            std::vector<float> logits(vocab.size() + 1 + durations.size());
            int hash = seed + frame * 13;
            for (int32_t token : history) hash = hash * 3 + token + 1;
            for (size_t j = 0; j < logits.size(); ++j)
                logits[j] = (float)((hash + (int)j * 7) % 17 - 8) / 5;
            rows.push_back({frame, history, std::move(logits)});
        }
        for (size_t token = 0; token < vocab.size(); ++token) {
            const std::string next = text + vocab[token];
            if (std::any_of(words.begin(), words.end(), [&](const std::string& word) {
                return word.compare(0, next.size(), next) == 0;
            })) {
                Ids next_ids = history;
                next_ids.push_back((int32_t)token);
                histories.push_back(std::move(next_ids));
            }
        }
    }
    Oracle oracle{rows, vocab, words, durations, T, {}};
    oracle.enumerate(0, {}, "");
    pk::TokenGrammar grammar(rules(words), vocab);
    for (bool norm : {false, true}) {
        auto got = pk::detail::tdt_beam_search_scored(
            rows, T, durations, (int)vocab.size(), 16, 16, norm, grammar);
        require(got.size() == oracle.probability.size(), "oracle hypothesis count");
        auto remaining = oracle.probability;
        float previous = std::numeric_limits<float>::infinity();
        for (const auto& hyp : got) {
            const auto path = ids(hyp);
            const float ranked = norm ? hyp.normalized_score : hyp.score;
            require(ranked <= previous, "oracle ranking");
            previous = ranked;
            require(std::fabs(hyp.normalized_score - hyp.score / (path.size() + 1)) < 1e-5,
                    "score normalization");
            auto found = std::find_if(remaining.begin(), remaining.end(), [&](const auto& entry) {
                return entry.first.first == path &&
                    std::fabs((double)std::log(entry.second) - hyp.score) < 2e-5;
            });
            require(found != remaining.end(), "oracle path score");
            remaining.erase(found);
            for (const auto& token : hyp.tokens) {
                require(token.frame >= 0 && token.frame < T, "token frame");
                require(std::find(durations.begin(), durations.end(), token.duration) != durations.end(),
                        "token duration");
            }
        }
    }
}

void terminal_eligibility() {
    const auto vocab = pieces(8);
    pk::TokenGrammar alternatives("root ::= \"a\" | \"bc\"", vocab);
    Rows labels{{0, {}, {4, 5, 0, 0, 0, 0, 0, 0, 10, 0}}};
    const auto label_result = pk::detail::tdt_beam_search_scored(
        labels, 1, {1}, 8, 1, 1, false, alternatives);
    const bool label_ok = label_result.size() == 1 && ids(label_result[0]) == Ids{0};
    if (!label_ok) std::fprintf(stderr, "terminal eligibility: beam-one label completion lost\n");

    pk::TokenGrammar ab("root ::= \"ab\"", vocab);
    Rows durations{{0, {}, {10, -30, -30, -30, -30, -30, -30, -30, -20, 0, 10}},
                   {0, {0}, {-30, 10, -30, -30, -30, -30, -30, -30, -20, 0, 10}}};
    const auto duration_result = pk::detail::tdt_beam_search_scored(
        durations, 1, {0, 1}, 8, 1, 1, false, ab);
    const bool duration_ok = duration_result.size() == 1 && ids(duration_result[0]) == Ids({0, 1}) &&
        duration_result[0].tokens[0].duration == 0 && duration_result[0].tokens[1].duration == 1;
    if (!duration_ok) std::fprintf(stderr, "terminal eligibility: beam-one zero duration lost\n");
    Rows short_rows{{0, {}, {10, -30, -30, -30, -30, -30, -30, -30, -20, 0, 10}},
                    {1, {0}, {-30, 10, -30, -30, -30, -30, -30, -30, -20, 0, 10}}};
    const auto short_result = pk::detail::tdt_beam_search_scored(
        short_rows, 2, {1, 2}, 8, 1, 1, false, ab);
    const bool short_ok = short_result.size() == 1 && ids(short_result[0]) == Ids({0, 1}) &&
        short_result[0].tokens[0].duration == 1 && short_result[0].tokens[1].frame == 1;
    if (!short_ok) std::fprintf(stderr, "terminal eligibility: beam-one short duration lost\n");

    Rows cross{{0, {}, {9, 10, -30, -30, -30, -30, -30, -30, -30, 0, 10}}};
    const auto cross_result = pk::detail::tdt_beam_search_scored(
        cross, 1, {0, 1}, 8, 1, 1, false, alternatives);
    const bool cross_ok = cross_result.size() == 1 && ids(cross_result[0]) == Ids{0} &&
        cross_result[0].tokens[0].duration == 1;
    if (!cross_ok) std::fprintf(stderr, "terminal eligibility: jointly eligible pair lost\n");

    Rows blanks{{0, {}, {-20, -30, -30, -30, -30, -30, -30, -30, 20, -20, 0, 20}},
                {1, {}, {10, -30, -30, -30, -30, -30, -30, -30, -20, 0, 5, 10}},
                {1, {0}, {-30, 10, -30, -30, -30, -30, -30, -30, -20, 0, 5, 10}}};
    const auto blank_result = pk::detail::tdt_beam_search_scored(
        blanks, 2, {0, 1, 2}, 8, 1, 1, false, ab);
    const bool blank_ok = blank_result.size() == 1 && ids(blank_result[0]) == Ids({0, 1}) &&
        blank_result[0].tokens[0].frame == 1 && blank_result[0].tokens[1].frame == 1;
    if (!blank_ok) std::fprintf(stderr, "terminal eligibility: viable blank duration lost\n");
    require(label_ok && duration_ok && short_ok && cross_ok && blank_ok,
            "terminal eligibility must precede candidate pruning");
    const auto logp = [](const std::vector<float>& logits, size_t first, size_t count, size_t chosen) {
        long double sum = 0;
        for (size_t i = first; i < first + count; ++i) sum += std::exp((long double)logits[i]);
        return (double)std::log(std::exp((long double)logits[chosen]) / sum);
    };
    require(std::fabs(label_result[0].score - logp(labels[0].logits, 0, 9, 0)) < 2e-5,
            "ineligible terminal labels retain their original probability mass");
    const double expected_duration_score = logp(durations[0].logits, 0, 9, 0) +
        logp(durations[0].logits, 9, 2, 9) + logp(durations[1].logits, 0, 9, 1) +
        logp(durations[1].logits, 9, 2, 10);
    require(std::fabs(duration_result[0].score - expected_duration_score) < 2e-5,
            "ineligible terminal durations retain their original probability mass");
}

void synthetic() {
    terminal_eligibility();
    for (int seed = 0; seed < 12; ++seed) {
        finite_oracle(pieces(), {"a", "ab", "b"}, {1, 2}, seed);
        finite_oracle(pieces(), {"a", "ab", "aa"}, {0, 1, 2}, seed);
        auto duplicates = pieces();
        duplicates[1] = "a";
        finite_oracle(duplicates, {"ac"}, {0, 1, 2}, seed);
    }

    const auto vocab = pieces(8);
    pk::TokenGrammar c("root ::= \"c\"", vocab);
    std::vector<float> logits{10, 9, -3, 8, 7, 6, 5, 4, 20, 0};
    Rows low{{0, {}, logits}};
    auto result = pk::detail::tdt_beam_search_scored(low, 1, {1}, 8, 1, 1, false, c);
    require(result.size() == 1 && ids(result[0]) == Ids{2}, "legal token below unconstrained top-k");
    long double denominator = 0;
    for (size_t i = 0; i < 9; ++i) denominator += std::exp((long double)logits[i]);
    require(std::fabs(result[0].score - (-3 - (double)std::log(denominator))) < 2e-5,
            "illegal tokens and blank remain in probability denominator");

    auto pair_rows = low;
    pair_rows[0].logits.back() = -2;
    pair_rows[0].logits.push_back(2);
    result = pk::detail::tdt_beam_search_scored(pair_rows, 1, {1, 2}, 8, 1, 1, false, c);
    require(result.size() == 1 && ids(result[0]) == Ids{2} && result[0].tokens[0].duration == 2,
            "legal token-duration pair survives pruning");
    require(std::fabs(result[0].score - (-3 - (double)std::log(denominator) - std::log1p(std::exp(-4.0)))) < 2e-5,
            "original duration probability");
    pk::TokenGrammar controls("root ::= \"c\"", {"x", "<EOU>", "<EOB>", "[noise]", "", "c", "a", "b"});
    result = pk::detail::tdt_beam_search_scored(low, 1, {1}, 8, 1, 1, false, controls);
    require(result.size() == 1 && ids(result[0]) == Ids{5}, "controls cannot become text");

    pk::TokenGrammar final("root ::= \"a\" | \"bc\"", vocab);
    Rows final_rows{{0, {}, {4, 5, 0, 0, 0, 0, 0, 0, 10, 0}}};
    result = pk::detail::tdt_beam_search_scored(final_rows, 1, {1}, 8, 2, 1, false, final);
    require(result.size() == 1 && ids(result[0]) == Ids{0}, "acceptance before final pruning");
    pk::TokenGrammar incomplete("root ::= \"ab\"", vocab);
    require(pk::detail::tdt_beam_search_scored(final_rows, 1, {1}, 8, 2, 1, true, incomplete).empty(),
            "incomplete path is not a match");
    require(pk::detail::tdt_beam_search_scored({}, 0, {1}, 8, 1, 1, true, c).empty(),
            "empty input nonnullable");
    pk::TokenGrammar empty("root ::= \"\"", vocab);
    result = pk::detail::tdt_beam_search_scored({}, 0, {1}, 8, 1, 1, true, empty);
    require(result.size() == 1 && result[0].tokens.empty() && result[0].score == 0,
            "empty input nullable");

    pk::TokenGrammar ab("root ::= \"ab\"", vocab);
    Rows blanks{{0, {}, {-30, -30, -30, -30, -30, -30, -30, -30, 10, 0}},
                {1, {}, {10, -30, -30, -30, -30, -30, -30, -30, -20, 0}},
                {2, {0}, {-30, 10, -30, -30, -30, -30, -30, -30, -20, 0}}};
    result = pk::detail::tdt_beam_search_scored(blanks, 3, {1}, 8, 1, 1, true, ab);
    require(result.size() == 1 && ids(result[0]) == Ids({0, 1}) &&
            result[0].tokens[0].frame == 1 && result[0].tokens[1].frame == 2,
            "blank preserves grammar and time");

    pk::TokenGrammar a("root ::= \"a\"", vocab);
    Rows fallback{{0, {}, {10, -30, -30, -30, -30, -30, -30, -30, -20, 10, 0}},
                  {0, {0}, {-30, -30, -30, -30, -30, -30, -30, -30, 10, 10, 0}},
                  {1, {0}, {-30, -30, -30, -30, -30, -30, -30, -30, 10, 10, 0}}};
    result = pk::detail::tdt_beam_search_scored(fallback, 2, {0, 1}, 8, 1, 1, false, a);
    require(result.size() == 1 && ids(result[0]) == Ids{0} && result[0].tokens[0].duration == 0 &&
            result[0].tokens[0].frame == 0 && result[0].score < -20,
            "blank zero-duration fallback uses scored positive durations");

    pk::TokenGrammar tie("root ::= \"a\" | \"b\"", vocab);
    Rows ties{{0, {}, {2, 2, 0, 0, 0, 0, 0, 0, 0, 0}}};
    for (int i = 0; i < 5; ++i) {
        result = pk::detail::tdt_beam_search_scored(ties, 1, {1}, 8, 2, 2, true, tie);
        require(result.size() == 2 && ids(result[0]) == Ids{0} && ids(result[1]) == Ids{1} &&
                result[0].score == result[1].score, "deterministic ties");
    }

    for (const std::string& piece : {std::string("a"), std::string("▁")}) {
        auto cycle_vocab = vocab;
        cycle_vocab[0] = piece;
        pk::TokenGrammar cycle(piece == "a" ? "root ::= \"a\"*" : "root ::= \" \"*", cycle_vocab);
        Rows cycles;
        for (int n = 0; n <= 128; ++n) {
            std::vector<float> scores(11, -1000);
            scores[0] = scores[9] = 1000; // Exact zero log-probability token/duration cycle.
            cycles.push_back({0, Ids((size_t)n, 0), std::move(scores)});
        }
        fails([&] { pk::detail::tdt_beam_search_scored(cycles, 1, {0, 1}, 8, 1, 1, true, cycle); },
              "symbol limit");
    }

    pk::detail::TdtSearchLimits limits;
    limits.max_work = 0;
    fails([&] { pk::detail::tdt_beam_search_scored(low, 1, {1}, 8, 1, 1, false, c, limits); }, "work limit");
    limits = {};
    limits.max_expansions = 0;
    fails([&] { pk::detail::tdt_beam_search_scored(low, 1, {1}, 8, 1, 1, false, c, limits); }, "expansion limit");
    limits = {};
    limits.max_hypotheses = 0;
    fails([&] { pk::detail::tdt_beam_search_scored(low, 1, {1}, 8, 1, 1, false, c, limits); }, "hypothesis limit");
    limits = {};
    limits.max_live_tokens = 0;
    fails([&] { pk::detail::tdt_beam_search_scored(low, 1, {1}, 8, 1, 1, false, c, limits); }, "history limit");
    limits = {};
    limits.max_tokens = 0;
    fails([&] { pk::detail::tdt_beam_search_scored(low, 1, {1}, 8, 1, 1, false, c, limits); }, "symbol limit");

    for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()}) {
        for (size_t index : {size_t(0), size_t(9)}) {
            Rows corrupt = low;
            corrupt[0].logits[index] = bad;
            fails([&] { pk::detail::tdt_beam_search_scored(corrupt, 1, {1}, 8, 1, 1, false, c); }, "non-finite");
        }
    }
    Rows overflow_pair = pair_rows;
    overflow_pair[0].logits[2] = -3e38f;
    overflow_pair[0].logits[9] = -3e38f;
    fails([&] { pk::detail::tdt_beam_search_scored(overflow_pair, 1, {1, 2}, 8, 2, 1, false, c); },
          "non-finite pair");
    pk::TokenGrammar repeated("root ::= \"ccc\"", vocab);
    Rows overflow_sum;
    for (int n = 0; n < 3; ++n) {
        std::vector<float> scores(11, 0);
        scores[2] = -1.2e38f;
        scores[10] = -3e38f;
        overflow_sum.push_back({0, Ids((size_t)n, 2), std::move(scores)});
    }
    fails([&] { pk::detail::tdt_beam_search_scored(overflow_sum, 1, {0, 1}, 8, 1, 1, false, repeated); },
          "non-finite accumulated");
    Rows wrong_shape = low;
    wrong_shape[0].logits.pop_back();
    fails([&] { pk::detail::tdt_beam_search_scored(wrong_shape, 1, {1}, 8, 1, 1, false, c); }, "logits shape");
    for (const std::vector<int32_t>& durations : {std::vector<int32_t>{}, {-1, 1}, {0}, {0, 0, 1}})
        fails([&] { pk::detail::tdt_beam_search_scored(low, 1, durations, 8, 1, 1, false, c); }, "duration");
    fails([&] { pk::detail::tdt_beam_search_scored(low, 1, {1}, 8, 0, 1, false, c); }, "beam_size");
    fails([&] { pk::detail::tdt_beam_search_scored(low, 1, {1}, 8, 65, 1, false, c); }, "beam exceeds");
    fails([&] { pk::detail::tdt_beam_search_scored(low, 1, {1}, 7, 1, 1, false, c); }, "vocabulary mismatch");
    result = pk::detail::tdt_beam_search_scored(low, 1, {INT32_MAX}, 8, 1, 1, false, c);
    require(result.size() == 1 && result[0].tokens[0].duration == INT32_MAX, "large positive duration");

    pk::TokenGrammar single("root ::= \"a\"", {"a"});
    result = pk::detail::tdt_beam_search_scored({{0, {}, {0, 0, 0}}}, 1, {1}, 1, 1, 1, false, single);
    require(result.size() == 1 && ids(result[0]) == Ids{0}, "single text label");
    std::fprintf(stderr, "test_tdt_grammar: synthetic oracle and boundary cases passed\n");
}

bool same(const std::vector<pk::TdtBeamHypothesis>& a,
          const std::vector<pk::TdtBeamHypothesis>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].score != b[i].score || a[i].normalized_score != b[i].normalized_score ||
            a[i].tokens.size() != b[i].tokens.size()) return false;
        for (size_t j = 0; j < a[i].tokens.size(); ++j) {
            const auto& x = a[i].tokens[j];
            const auto& y = b[i].tokens[j];
            if (x.id != y.id || x.frame != y.frame || x.duration != y.duration) return false;
        }
    }
    return true;
}

int model_test() {
    const char* path = std::getenv("PARAKEET_TEST_GGUF");
    if (!path) {
        std::fprintf(stderr, "test_tdt_grammar_model: PARAKEET_TEST_GGUF not set; skip\n");
        return 77;
    }
    struct Shutdown { ~Shutdown() { pk::shutdown_backend(); } } shutdown;
    pk::ModelLoader loader;
    require(loader.load(path), "load model");
    const auto& cfg = loader.config();
    require(!cfg.tdt_durations.empty(), "test requires TDT model");
    pk::Audio audio;
    require(pk::load_audio_16k_mono("tests/fixtures/speech.wav", audio), "load fixture");
    pk::MelFrontend mel(loader);
    int nm = 0, frames = 0;
    std::vector<float> features;
    mel.compute(audio.samples, features, nm, frames);
    pk::Encoder encoder(loader);
    std::vector<float> channels;
    int hidden = 0, T = 0;
    encoder.forward(features, nm, frames, channels, hidden, T);
    std::vector<float> enc(channels.size());
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < hidden; ++c) enc[(size_t)t * hidden + c] = channels[(size_t)c * T + t];
    pk::PredictionNet pred(loader);
    pk::Joint joint(loader);
    const auto plain = pk::tdt_beam_search(pred, joint, enc, T, hidden, cfg.tdt_durations,
                                          (int)cfg.blank_id, 4, 4);
    const std::string expected = "Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait.";
    require(!plain.empty() && pk::detokenize(loader.tokenizer_pieces(), ids(plain[0])) == expected,
            "unconstrained beam reference transcript");
    for (const std::string& text : {expected, std::string("hello")}) {
        pk::TokenGrammar grammar(rules({text}), loader.tokenizer_pieces());
        const auto got = pk::tdt_beam_search(pred, joint, enc, T, hidden, cfg.tdt_durations,
                                           (int)cfg.blank_id, 4, 1, true, &grammar);
        require(got.size() == 1 && pk::detokenize(loader.tokenizer_pieces(), ids(got[0])) == text &&
                std::isfinite(got[0].score), "constrained model text");
        const auto again = pk::tdt_beam_search(pred, joint, enc, T, hidden, cfg.tdt_durations,
                                             (int)cfg.blank_id, 4, 1, true, &grammar);
        require(same(got, again), "deterministic shared grammar reuse");
    }
    const auto after = pk::tdt_beam_search(pred, joint, enc, T, hidden, cfg.tdt_durations,
                                          (int)cfg.blank_id, 4, 4, true, nullptr);
    require(same(plain, after), "null grammar remains unchanged after constrained use");
    pk::TokenGrammar grammar(rules({"hello"}), loader.tokenizer_pieces());
    fails([&] { pk::tdt_beam_search(pred, joint, enc, T, hidden, {1}, (int)cfg.blank_id,
                                   4, 1, true, &grammar); }, "incompatible TDT head");
    fails([&] { pk::tdt_beam_search(pred, joint, {}, T, hidden, cfg.tdt_durations,
                                   (int)cfg.blank_id, 4, 1, true, &grammar); }, "encoder shape");
    std::fprintf(stderr, "test_tdt_grammar_model: 110m natural/forced text, reuse and null grammar passed\n");
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--model") return model_test();
        require(argc == 1, "unexpected test arguments");
        synthetic();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_tdt_grammar: %s\n", e.what());
        return 1;
    }
}
