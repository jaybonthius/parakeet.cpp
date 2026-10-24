#include "grammar.hpp"
#include "tokenizer.hpp"

#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void rejects(const std::function<void()>& action, const char* diagnostic) {
    try { action(); }
    catch (const std::runtime_error& e) {
        require(std::string(e.what()).find(diagnostic) != std::string::npos, e.what());
        return;
    }
    throw std::runtime_error("expected an error");
}
bool accepts(const pk::TokenGrammar& grammar, const std::vector<int32_t>& tokens) {
    int state = grammar.start();
    for (int token : tokens) {
        state = grammar.transition(state, token);
        if (state < 0) return false;
    }
    return grammar.accepting(state);
}

// Enumerate token sequences, not CTC alignments. Compare with an independent
// rendered-text predicate, preserving every compatible token segmentation.
void oracle(const std::string& rules, const std::vector<std::string>& pieces, int depth,
            const std::function<bool(const std::string&)>& in_language) {
    pk::TokenGrammar grammar(rules, pieces);
    require(grammar.token_count() == static_cast<int>(pieces.size()), "vocabulary size changed");
    for (size_t q = 0; q < grammar.state_count(); ++q) {
        int previous = -1;
        for (auto arc : grammar.arcs(static_cast<int>(q))) {
            require(arc.token > previous, "arcs are not uniquely token-sorted");
            require(arc.next >= 0 && static_cast<size_t>(arc.next) < grammar.state_count(), "invalid target");
            require(grammar.transition(static_cast<int>(q), arc.token) == arc.next, "transition differs from arc");
            previous = arc.token;
        }
        require(grammar.transition(static_cast<int>(q), -1) == -1, "negative token allowed");
        require(grammar.transition(static_cast<int>(q), grammar.token_count()) == -1, "blank became text");
    }
    std::vector<int32_t> tokens;
    std::function<void(int, bool)> visit = [&](int remaining, bool text_only) {
        require(accepts(grammar, tokens) == (text_only && in_language(pk::detokenize(pieces, tokens))),
                "token oracle disagrees with rendered language");
        if (!remaining) return;
        for (size_t t = 0; t < pieces.size(); ++t) {
            const auto& piece = pieces[t];
            bool is_text = !piece.empty() &&
                !((piece.front() == '<' && piece.back() == '>') ||
                  (piece.front() == '[' && piece.back() == ']'));
            tokens.push_back(static_cast<int32_t>(t));
            visit(remaining - 1, text_only && is_text);
            tokens.pop_back();
        }
    };
    visit(depth, true);
}
} // namespace

int main() {
    try {
        const std::vector<std::string> pieces{"a", "b", "ab", "aa", "▁", "▁a", " a", "  a", "é", "😀",
                                               "<EOU>", "<EOB>", "<unk>", "[noise]", ""};
        oracle("root ::= (\"ab\" | \"aa\" | \" a\" | \"é😀\" | \"\")", pieces, 4,
               [](const std::string& s) { return s == "ab" || s == "aa" || s == " a" || s == "é😀" || s.empty(); });
        oracle("root ::= word (\"b\" | \"a\")*\nword ::= \"a\" # reference\n", pieces, 3,
               [](const std::string& s) { return !s.empty() && s[0] == 'a' && s.find_first_not_of("ab") == std::string::npos; });
        oracle("root ::= [é-ê]+ [^a-béê ]?", {"é", "ê", "a", "b", "x", "😀", "▁"}, 4,
               [](const std::string& s) {
                   size_t i = 0;
                   while (s.compare(i, 2, "é") == 0 || s.compare(i, 2, "ê") == 0) i += 2;
                   return i > 0 && (i == s.size() || s.substr(i) == "x" || s.substr(i) == "😀");
               });
        oracle("root ::= (\n \"a\" | # comment\n \"b\"\n)?", {"a", "b", "ab", "▁"}, 4,
               [](const std::string& s) { return s.empty() || s == "a" || s == "b"; });
        oracle("root ::= (\"ab\" | \"a\") \"b\"?", {"a", "b", "ab", "abb"}, 4,
               [](const std::string& s) { return s == "a" || s == "ab" || s == "abb"; });
        oracle("root ::= (\"\" | \"a\")*", {"a", "aa", "▁", "b"}, 4,
               [](const std::string& s) { return s.find_first_not_of('a') == std::string::npos; });

        pk::TokenGrammar repeated("root ::= \"aa\"", {"a"});
        require(accepts(repeated, {0, 0}) && !accepts(repeated, {0}), "shared grammar collapsed repeated labels");
        pk::TokenGrammar spaces("root ::= \" a\"", {"▁", "▁a", "  a"});
        require(accepts(spaces, {0, 1}) && accepts(spaces, {2}), "single space trim lost segmentation");
        require(!accepts(spaces, {1}) && !accepts(spaces, {0, 0, 1}), "space trim repeated or omitted");
        pk::TokenGrammar unit("root ::= \"a\"", {"▁a"});
        int complete = unit.transition(unit.start(), 0);
        require(unit.accepting(complete) && unit.transition(complete, 0) == -1, "invalid unit ending");
        for (int i = 0; i < 3; ++i) {
            int restarted = unit.start();
            require(unit.accepting(unit.transition(restarted, 0)), "restart did not restore initial trim");
        }
        pk::TokenGrammar empty("root ::= \"\"", {"▁", ""});
        require(empty.accepting(empty.start()) && accepts(empty, {0}), "empty language/first space failed");
        require(!accepts(empty, {0, 0}) && !accepts(empty, {1}), "empty pieces or extra trim allowed");
        pk::TokenGrammar escaped("root ::= \"\\u00e9\\U0001f600\\x21\\n\\t\\r\\\"\\\\\"", {"é😀!\n\t\r\"\\"});
        require(accepts(escaped, {0}), "escaped Unicode/ASCII failed");
        pk::TokenGrammar controls("root ::= \"<EOU>\" | \"<EOB>\" | \"[noise]\" | \"<unk>\"",
                                  {"<EOU>", "<EOB>", "[noise]", "<unk>"});
        require(controls.arcs(controls.start()).empty(), "special tokens became text arcs");
        pk::TokenGrammar unavailable("root ::= \"é\"", {"a"});
        require(!unavailable.accepting(unavailable.start()) && unavailable.arcs(unavailable.start()).empty(),
                "unrepresentable language accepted");
        std::string source = "root ::= \"a\"";
        std::vector<std::string> vocabulary{"a"};
        pk::TokenGrammar owned(source, vocabulary);
        source.clear();
        vocabulary.clear();
        require(accepts(owned, {0}), "compiled grammar borrowed caller buffers");
        pk::TokenGrammar dense("root ::= \"a\"*", std::vector<std::string>(4096, "a"));
        require(dense.state_count() <= 2 && dense.arcs(dense.start()).size() == 4096,
                "shared compiler expanded token states into a CTC product");

        for (const char* bad : {"", "other ::= \"a\"", "root := \"a\"", "root ::= unknown", "root ::= root",
                "root ::= x\nx ::= root", "root ::= \"a\"{2}", "root ::= .", "root ::= []", "root ::= [z-a]",
                "root ::= (\"a\"", "root ::= \"unfinished", "root ::= \"\\q\"", "root ::= \"\\uD800\"",
                "root ::= \"\\U00110000\"", "root ::= \"\\x00\"", "root ::= \"a\"**", "root ::= \"a\"\nroot ::= \"b\""})
            rejects([&] { pk::TokenGrammar g(bad, {"a"}); }, "grammar:");
        for (const std::string& invalid : {std::string(1, char(0xff)), std::string("\xc0\x80"),
                std::string("\xed\xa0\x80"), std::string("\xf4\x90\x80\x80"), std::string("\xe2\x96"), std::string(1, '\0')}) {
            rejects([&] { pk::TokenGrammar g("root ::= \"" + invalid + "\"", {"a"}); }, "grammar:");
            rejects([&] { pk::TokenGrammar g("root ::= \"a\"", {invalid}); }, "grammar:");
        }
        rejects([&] { pk::TokenGrammar g("root ::= \"a\"", {}); }, "vocabulary");
        rejects([&] { pk::TokenGrammar g("root ::= \"a\"", std::vector<std::string>(65537, "a")); }, "vocabulary");
        rejects([&] { pk::TokenGrammar g(std::string(65537, ' '), {"a"}); }, "source limit");
        rejects([&] { pk::TokenGrammar g("root ::= " + std::string(65, '(') + "\"a\"" + std::string(65, ')'), {"a"}); }, "limit");
        rejects([&] { pk::TokenGrammar g("root ::= \"" + std::string(4100, 'a') + "\"", {"a"}); }, "syntax node limit");
        rejects([&] { pk::TokenGrammar g("root ::= \"" + std::string(2200, 'a') + "\"", {"a"}); }, "NFA state limit");
        std::string exponential = "root ::= [ab]* \"a\"";
        for (int i = 0; i < 12; ++i) exponential += " [ab]";
        rejects([&] { pk::TokenGrammar g(exponential, {"a", "b"}); }, "token state limit");
        rejects([&] { pk::TokenGrammar g("root ::= \"aaa\"", std::vector<std::string>(65536, "a")); }, "token arc limit");
        std::string wide = "root ::= ";
        for (int i = 0; i < 400; ++i) wide += "\"a\"? ";
        rejects([&] { pk::TokenGrammar g(wide, std::vector<std::string>(65536, "b")); }, "compilation work limit");
        std::puts("test_grammar: PASS (token oracle, restart, Unicode/space/segmentation, controls, failures/limits)");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_grammar: %s\n", e.what());
        return 1;
    }
}
