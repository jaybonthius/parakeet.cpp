#include "ctc_grammar.hpp"
#include "tokenizer.hpp"

#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void rejects(const std::function<void()>& action, const char* diagnostic = nullptr) {
    try { action(); }
    catch (const std::runtime_error& e) {
        if (diagnostic) require(std::string(e.what()).find(diagnostic) != std::string::npos, e.what());
        return;
    }
    throw std::runtime_error("expected an error");
}

// Enumerate raw alignments, independently collapse them, render with the existing
// tokenizer, then apply a text-language predicate. No compiled grammar is used.
void oracle(const std::string& rules, const std::vector<std::string>& pieces, int frames,
            const std::function<bool(const std::string&)>& accepts, unsigned seed) {
    int blank = static_cast<int>(pieces.size()), classes = blank + 1;
    pk::CtcGrammar grammar(rules, pieces, blank);
    std::mt19937 rng(seed);
    std::vector<float> probs(static_cast<size_t>(frames) * classes);
    for (float& p : probs) p = -static_cast<float>(rng() % 7); // includes exact ties
    if (!probs.empty()) probs[rng() % probs.size()] = -std::numeric_limits<float>::infinity();
    double best = -std::numeric_limits<double>::infinity();
    std::vector<int> raw(static_cast<size_t>(frames));
    std::function<void(int, double)> visit = [&](int t, double score) {
        if (t < frames) {
            for (int token = 0; token < classes; ++token) {
                raw[t] = token;
                visit(t + 1, score + probs[static_cast<size_t>(t) * classes + token]);
            }
            return;
        }
        std::vector<int32_t> tokens;
        int previous = blank;
        for (int token : raw) {
            if (token != previous && token != blank) {
                const auto& piece = pieces[token];
                if (piece.empty() || piece.front() == '<' || piece.front() == '[') return;
                tokens.push_back(token);
            }
            previous = token;
        }
        if (accepts(pk::detokenize(pieces, tokens)) && score > best) best = score;
    };
    visit(0, 0);
    auto result = grammar.decode(probs, frames, classes);
    require(result.matched == std::isfinite(best), "oracle match status differs");
    if (result.matched) {
        require(std::abs(result.score - best) < 1e-9, "oracle best alignment score differs");
        require(accepts(pk::detokenize(pieces, result.tokens)), "result is not in text language");
        require(grammar.decode(probs, frames, classes).tokens == result.tokens, "unstable tie handling");
    }
}
}

int main() {
    try {
        const std::vector<std::string> pieces{"▁a", "a", "b", "ab", "▁", "é", "x", "<ctrl>", "[noise]", ""};
        for (unsigned seed = 0; seed < 16; ++seed) {
            oracle("root ::= word (\"b\" | \"a\")*\nword ::= \"a\" # reference\n", pieces, 4,
                   [](const std::string& s) {
                       return !s.empty() && s[0] == 'a' && s.find_first_not_of("ab") == std::string::npos;
                   }, seed);
            oracle("root ::= (\"ab\" | \"é\" | \" a\" | \"\")", pieces, 3,
                   [](const std::string& s) { return s == "ab" || s == "é" || s == " a" || s.empty(); }, seed);
            oracle("root ::= [a-b]+ [^a-béx ]?", pieces, 3,
                   [](const std::string& s) { return !s.empty() && s.find_first_not_of("ab") == std::string::npos; }, seed);
        }
        oracle("root ::= \"\"", pieces, 0, [](const std::string& s) { return s.empty(); }, 0);
        oracle("root ::= \"a\"", pieces, 0, [](const std::string& s) { return s == "a"; }, 0);
        const std::vector<std::string> tiny{"a", "b"};
        const float neg = -std::numeric_limits<float>::infinity();
        pk::CtcGrammar repeated("root ::= \"aa\"", tiny, 2);
        require(!repeated.decode({0, neg, neg, 0, neg, neg}, 2, 3).matched, "repeat advanced grammar without blank");
        auto aa = repeated.decode({0, neg, neg, neg, neg, 0, 0, neg, neg}, 3, 3);
        require(aa.matched && aa.tokens == std::vector<int32_t>({0, 0}), "blank-separated repeat failed");
        auto shared = std::make_shared<const pk::TokenGrammar>("root ::= \"a\"", tiny);
        pk::CtcGrammar retained(shared, 2);
        shared.reset();
        require(retained.decode({0, neg, neg}, 1, 3).tokens == std::vector<int32_t>({0}),
                "CTC adapter did not retain shared grammar");
        rejects([&] { pk::CtcGrammar g(std::shared_ptr<const pk::TokenGrammar>{}, 2); }, "layout");
        pk::CtcGrammar a("root ::= \"a\"", tiny, 2);
        require(a.decode({-1000, 0, neg}, 1, 3).tokens == std::vector<int32_t>({0}), "illegal high score not excluded");
        require(!a.decode({neg, 0, neg}, 1, 3).matched, "impossible path returned success");
        require(!repeated.decode({0, neg, neg}, 1, 3).matched, "partial accepted as complete");
        pk::CtcGrammar escaped("root ::= \"\\u00e9\\U0001f600\\x21\\n\\t\\r\\\"\\\\\"", {"é😀!\n\t\r\"\\"}, 1);
        require(escaped.decode({0, neg}, 1, 2).matched, "Unicode/ASCII escapes failed");
        pk::CtcGrammar unicode("root ::= [é-ê]+", {"é", "ê"}, 2);
        require(unicode.decode({0, neg, neg, neg, 0, neg}, 2, 3).matched, "Unicode range failed");
        pk::CtcGrammar negated("root ::= [^a]", {"a", "é"}, 2);
        require(negated.decode({neg, 0, neg}, 1, 3).matched, "negated Unicode class failed");
        pk::CtcGrammar control("root ::= \"<ctrl>\"", {"<ctrl>"}, 1);
        require(!control.decode({0, neg}, 1, 2).matched, "control token became text");
        pk::CtcGrammar unavailable("root ::= \"é\"", tiny, 2);
        require(!unavailable.decode({0, 0, 0}, 1, 3).matched, "unrepresentable text accepted");
        pk::CtcGrammar split("root ::= \"ab\"", {"a", "b", "ab"}, 3);
        require(split.decode({0, neg, neg, neg, neg, 0, neg, neg}, 2, 4).tokens == std::vector<int32_t>({0, 1}), "split segmentation failed");
        require(split.decode({neg, neg, 0, neg}, 1, 4).tokens == std::vector<int32_t>({2}), "whole-piece segmentation failed");
        pk::CtcGrammar spaces("root ::= \" a\"", {"▁", "▁a"}, 2);
        require(spaces.decode({0, neg, neg, neg, 0, neg}, 2, 3).matched, "initial-space removal across pieces failed");
        require(!spaces.decode({neg, 0, neg}, 1, 3).matched, "incorrect initial space retained");
        pk::CtcGrammar empty("root ::= \"\"", {"▁"}, 1);
        require(empty.decode({0, neg}, 1, 2).matched, "space-only first token must render empty");
        require(!empty.decode({0, neg, neg, 0, 0, neg}, 3, 2).matched, "more than one leading space stripped");
        pk::CtcGrammar multiline("root ::= (\n \"a\" | # comment\n \"b\"\n)?", tiny, 2);
        require(multiline.decode({}, 0, 3).matched, "multiline optional group failed");

        for (const char* bad : {"", "other ::= \"a\"", "root := \"a\"", "root ::= unknown",
                "root ::= root", "root ::= x\nx ::= root", "root ::= \"a\"{2}", "root ::= .",
                "root ::= []", "root ::= [z-a]", "root ::= (\"a\"", "root ::= \"unfinished",
                "root ::= \"\\q\"", "root ::= \"\\uD800\"", "root ::= \"\\U00110000\"",
                "root ::= \"\\x00\"", "root ::= \"a\"**", "root ::= \"a\"\nroot ::= \"b\""}) {
            rejects([&] { pk::CtcGrammar g(bad, tiny, 2); });
        }
        rejects([&] { pk::CtcGrammar g(std::string("root ::= \"") + char(0xc0) + char(0x80) + "\"", tiny, 2); }, "UTF-8");
        rejects([&] { pk::CtcGrammar g("root ::= \"a\"", {std::string(1, char(0xff))}, 1); }, "UTF-8");
        rejects([&] { pk::CtcGrammar g("root ::= \"a\"", tiny, 0); }, "layout");
        rejects([&] { pk::CtcGrammar g(std::string(65537, ' '), tiny, 2); }, "source limit");
        rejects([&] { pk::CtcGrammar g("root ::= " + std::string(65, '(') + "\"a\"" + std::string(65, ')'), tiny, 2); }, "limit");
        rejects([&] { pk::CtcGrammar g("root ::= \"" + std::string(2200, 'a') + "\"", tiny, 2); }, "NFA state limit");
        rejects([&] { a.decode({0}, 1, 3); }, "dimensions");
        rejects([&] { a.decode({}, -1, 3); }, "dimensions");
        for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), 0.1f})
            rejects([&] { a.decode({bad, 0, 0}, 1, 3); }, "value");
        std::string exponential = "root ::= [ab]* \"a\"";
        for (int i = 0; i < 12; ++i) exponential += " [ab]";
        rejects([&] { pk::CtcGrammar g(exponential, tiny, 2); }, "limit");
        std::vector<std::string> dense(100, "a");
        pk::CtcGrammar many("root ::= \"a\"*", dense, 100);
        rejects([&] { many.decode(std::vector<float>(101 * 10001, -1), 10001, 101); }, "work limit");
        rejects([&] { many.decode(std::vector<float>(101 * 170000, -1), 170000, 101); }, "traceback");
        std::puts("test_ctc_grammar: PASS (independent exhaustive oracle, text/token/CTC semantics, failures)");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_ctc_grammar: %s\n", e.what());
        return 1;
    }
}
