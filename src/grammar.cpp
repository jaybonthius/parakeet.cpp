#include "grammar.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>

namespace pk {
namespace {
constexpr size_t kMaxPieces = 65536;
constexpr size_t kMaxNodes = 4096;
constexpr size_t kMaxDepth = 64;
constexpr size_t kMaxTokenStates = 2048;
constexpr size_t kMaxTokenArcs = 131072;
constexpr size_t kMaxCompileWork = 20000000;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error("grammar: " + message);
}

uint32_t scalar(const std::string& s, size_t& pos) {
    if (pos == s.size()) fail("incomplete UTF-8");
    unsigned char c = s[pos++];
    if (c < 0x80) {
        if (c == 0) fail("NUL is not supported");
        return c;
    }
    int n = c >= 0xc2 && c <= 0xdf ? 1 : c >= 0xe0 && c <= 0xef ? 2 :
            c >= 0xf0 && c <= 0xf4 ? 3 : -1;
    if (n < 0) fail("invalid UTF-8");
    uint32_t value = c & ((1u << (6 - n)) - 1);
    for (int i = 0; i < n; ++i) {
        if (pos == s.size()) fail("incomplete UTF-8");
        unsigned char b = s[pos++];
        if ((b & 0xc0) != 0x80) fail("invalid UTF-8 continuation");
        value = (value << 6) | (b & 0x3f);
    }
    if (value < (n == 1 ? 0x80u : n == 2 ? 0x800u : 0x10000u) ||
        value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff)) fail("invalid Unicode scalar");
    return value;
}

struct Node {
    enum Kind { Chars, Sequence, Alternative, Reference, Optional, Star, Plus } kind;
    explicit Node(Kind value) : kind(value) {}
    std::vector<int> children;
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    bool negated = false;
    std::string name;
};

class Parser {
public:
    explicit Parser(const std::string& source) : source_(source) {
        if (source.size() > TokenGrammar::kMaxRuleBytes) fail("source limit exceeded");
        for (size_t p = 0; p < source.size();) scalar(source, p);
        space(true);
        while (pos_ < source_.size()) {
            std::string name = identifier();
            space(false);
            if (source_.compare(pos_, 3, "::=") != 0) error("expected ::=");
            pos_ += 3;
            int body = expression(false, 0);
            if (!rules.emplace(name, body).second) error("duplicate rule " + name);
            if (pos_ < source_.size() && source_[pos_] != '\n' && source_[pos_] != '\r')
                error("unsupported syntax");
            space(true);
        }
        if (!rules.count("root")) error("missing root rule");
        std::map<std::string, int> colors;
        for (const auto& rule : rules) check_rule(rule.first, colors, 0);
    }
    std::vector<Node> nodes;
    std::map<std::string, int> rules;
private:
    const std::string& source_;
    size_t pos_ = 0;
    [[noreturn]] void error(const std::string& message) const {
        fail(message + " at byte " + std::to_string(pos_));
    }
    void space(bool newline) {
        while (pos_ < source_.size()) {
            char c = source_[pos_];
            if (c == '#') {
                while (pos_ < source_.size() && source_[pos_] != '\n' && source_[pos_] != '\r') ++pos_;
            } else if (c == ' ' || c == '\t' || (newline && (c == '\n' || c == '\r'))) ++pos_;
            else break;
        }
    }
    static bool name_char(char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '-';
    }
    std::string identifier() {
        size_t start = pos_;
        while (pos_ < source_.size() && name_char(source_[pos_])) ++pos_;
        if (start == pos_) error("expected rule name");
        return source_.substr(start, pos_ - start);
    }
    int add(Node node) {
        if (nodes.size() == kMaxNodes) error("syntax node limit exceeded");
        nodes.push_back(std::move(node));
        return static_cast<int>(nodes.size() - 1);
    }
    uint32_t character() {
        if (pos_ == source_.size()) error("unterminated literal/class");
        if (source_[pos_] == '\n' || source_[pos_] == '\r') error("unescaped newline in literal/class");
        if (source_[pos_] != '\\') return scalar(source_, pos_);
        if (++pos_ == source_.size()) error("incomplete escape");
        char c = source_[pos_++];
        if (c == 'n') return '\n';
        if (c == 'r') return '\r';
        if (c == 't') return '\t';
        if (c == '\\' || c == '"' || c == '[' || c == ']' || c == '-') return c;
        int digits = c == 'x' ? 2 : c == 'u' ? 4 : c == 'U' ? 8 : 0;
        if (!digits) error("unsupported escape");
        uint32_t value = 0;
        for (int i = 0; i < digits; ++i) {
            if (pos_ == source_.size()) error("incomplete hexadecimal escape");
            char h = source_[pos_++];
            int d = h >= '0' && h <= '9' ? h - '0' : h >= 'a' && h <= 'f' ? h - 'a' + 10 :
                    h >= 'A' && h <= 'F' ? h - 'A' + 10 : -1;
            if (d < 0) error("invalid hexadecimal escape");
            value = (value << 4) | static_cast<uint32_t>(d);
        }
        if (value == 0 || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff))
            error("invalid Unicode scalar escape");
        return value;
    }
    int expression(bool nested, size_t depth) {
        if (depth > kMaxDepth) error("nesting limit exceeded");
        Node alt{Node::Alternative};
        do {
            if (!alt.children.empty()) ++pos_; // '|'
            Node seq{Node::Sequence};
            space(nested);
            while (pos_ < source_.size()) {
                char c = source_[pos_];
                if (c == '|' || c == ')' || c == '\r' || c == '\n') break;
                int atom;
                if (c == '"') {
                    ++pos_;
                    Node literal{Node::Sequence};
                    while (pos_ < source_.size() && source_[pos_] != '"') {
                        uint32_t cp = character();
                        Node ch{Node::Chars}; ch.ranges.emplace_back(cp, cp);
                        literal.children.push_back(add(std::move(ch)));
                    }
                    if (pos_ == source_.size()) error("unterminated literal");
                    ++pos_;
                    atom = add(std::move(literal));
                } else if (c == '[') {
                    ++pos_;
                    Node ch{Node::Chars};
                    if (pos_ < source_.size() && source_[pos_] == '^') { ch.negated = true; ++pos_; }
                    while (pos_ < source_.size() && source_[pos_] != ']') {
                        uint32_t lo = character(), hi = lo;
                        if (pos_ + 1 < source_.size() && source_[pos_] == '-' && source_[pos_ + 1] != ']') {
                            ++pos_; hi = character();
                        }
                        if (hi < lo) error("reversed character range");
                        ch.ranges.emplace_back(lo, hi);
                    }
                    if (pos_ == source_.size() || ch.ranges.empty()) error("unterminated/empty class");
                    ++pos_;
                    atom = add(std::move(ch));
                } else if (c == '(') {
                    ++pos_; atom = expression(true, depth + 1);
                    if (pos_ == source_.size() || source_[pos_] != ')') error("expected )");
                    ++pos_;
                } else if (name_char(c)) {
                    Node ref{Node::Reference}; ref.name = identifier(); atom = add(std::move(ref));
                } else error("unsupported syntax");
                space(nested);
                if (pos_ < source_.size() && (source_[pos_] == '?' || source_[pos_] == '*' || source_[pos_] == '+')) {
                    char op = source_[pos_++];
                    Node repeat{op == '?' ? Node::Optional : op == '*' ? Node::Star : Node::Plus};
                    repeat.children.push_back(atom); atom = add(std::move(repeat));
                    space(nested);
                }
                seq.children.push_back(atom);
            }
            alt.children.push_back(add(std::move(seq)));
        } while (pos_ < source_.size() && source_[pos_] == '|');
        return add(std::move(alt));
    }
    void check_node(int id, std::map<std::string, int>& colors, size_t depth) {
        if (depth > kMaxDepth) error("rule expansion depth limit exceeded");
        const Node& n = nodes[id];
        if (n.kind == Node::Reference) check_rule(n.name, colors, depth + 1);
        for (int child : n.children) check_node(child, colors, depth + 1);
    }
    void check_rule(const std::string& name, std::map<std::string, int>& colors, size_t depth) {
        auto it = rules.find(name);
        if (it == rules.end()) error("undefined rule " + name);
        if (colors[name] == 1) error("recursive rules are unsupported: " + name);
        if (colors[name] == 2) return;
        colors[name] = 1;
        check_node(it->second, colors, depth);
        colors[name] = 2;
    }
};

struct NfaState {
    std::vector<int> epsilon;
    int character = -1;
    int next = -1;
};

class Nfa {
public:
    explicit Nfa(const Parser& p) : parser(p) { ends = build(p.rules.at("root"), 0); }
    const Parser& parser;
    std::vector<NfaState> states;
    std::pair<int, int> ends;
    std::vector<int> closure(std::vector<int> ids, size_t& work) const {
        if (ids.empty()) return ids;
        std::vector<bool> seen(states.size(), false);
        for (int id : ids) seen[id] = true;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (++work > kMaxCompileWork) fail("compilation work limit exceeded");
            for (int next : states[ids[i]].epsilon) {
                if (++work > kMaxCompileWork) fail("compilation work limit exceeded");
                if (!seen[next]) { seen[next] = true; ids.push_back(next); }
            }
        }
        std::sort(ids.begin(), ids.end());
        return ids;
    }
    std::vector<int> step(const std::vector<int>& ids, uint32_t cp, size_t& work) const {
        std::vector<int> next;
        for (int id : ids) {
            if (++work > kMaxCompileWork) fail("compilation work limit exceeded");
            const auto& state = states[id];
            if (state.character < 0) continue;
            const Node& ch = parser.nodes[state.character];
            bool inside = false;
            for (auto range : ch.ranges) {
                if (++work > kMaxCompileWork) fail("compilation work limit exceeded");
                if (cp >= range.first && cp <= range.second) { inside = true; break; }
            }
            if (inside != ch.negated) next.push_back(state.next);
        }
        std::sort(next.begin(), next.end());
        next.erase(std::unique(next.begin(), next.end()), next.end());
        return closure(std::move(next), work);
    }
private:
    int state() {
        if (states.size() == kMaxNodes) fail("NFA state limit exceeded");
        states.emplace_back(); return static_cast<int>(states.size() - 1);
    }
    std::pair<int, int> build(int id, size_t depth) {
        if (depth > kMaxDepth) fail("rule expansion depth limit exceeded");
        const Node& n = parser.nodes[id];
        int start = state(), end = state();
        if (n.kind == Node::Chars) { states[start].character = id; states[start].next = end; }
        else if (n.kind == Node::Reference) {
            auto child = build(parser.rules.at(n.name), depth + 1);
            states[start].epsilon.push_back(child.first); states[child.second].epsilon.push_back(end);
        } else if (n.kind == Node::Sequence) {
            int prev = start;
            for (int child_id : n.children) {
                auto child = build(child_id, depth + 1);
                states[prev].epsilon.push_back(child.first); prev = child.second;
            }
            states[prev].epsilon.push_back(end);
        } else if (n.kind == Node::Alternative) {
            for (int child_id : n.children) {
                auto child = build(child_id, depth + 1);
                states[start].epsilon.push_back(child.first); states[child.second].epsilon.push_back(end);
            }
        } else {
            auto child = build(n.children[0], depth + 1);
            states[start].epsilon.push_back(child.first); states[child.second].epsilon.push_back(end);
            if (n.kind != Node::Plus) states[start].epsilon.push_back(end);
            if (n.kind != Node::Optional) states[child.second].epsilon.push_back(child.first);
        }
        return {start, end};
    }
};
} // namespace

TokenGrammar::TokenGrammar(const std::string& rules, const std::vector<std::string>& pieces)
    : tokens_(0) {
    if (pieces.empty() || pieces.size() > kMaxPieces) fail("unsupported vocabulary size");
    tokens_ = static_cast<int>(pieces.size());
    Parser parser(rules);
    Nfa nfa(parser);
    std::vector<std::vector<uint32_t>> rendered(pieces.size());
    for (size_t token = 0; token < pieces.size(); ++token) {
        const auto& piece = pieces[token];
        if (piece.empty() || ((piece.front() == '<' && piece.back() == '>') ||
                             (piece.front() == '[' && piece.back() == ']'))) continue;
        // Require complete UTF-8 pieces: byte-fallback and cross-piece scalars are not supported.
        for (size_t p = 0; p < piece.size();) {
            uint32_t cp = scalar(piece, p);
            rendered[token].push_back(cp == 0x2581 ? ' ' : cp);
        }
    }
    using Key = std::pair<bool, std::vector<int>>;
    size_t arcs = 0, work = 0;
    std::vector<Key> keys{{true, nfa.closure({nfa.ends.first}, work)}};
    std::map<Key, int> indices{{keys[0], 0}};
    states_.emplace_back();
    for (size_t q = 0; q < keys.size(); ++q) {
        Key key = keys[q];
        states_[q].accepting = std::binary_search(key.second.begin(), key.second.end(), nfa.ends.second);
        for (size_t token = 0; token < rendered.size(); ++token) {
            if (++work > kMaxCompileWork) fail("compilation work limit exceeded");
            if (rendered[token].empty()) continue;
            auto subset = key.second;
            const auto& chars = rendered[token];
            // Strip only the first character of the concatenated rendered text, if it is a space.
            size_t first = key.first && chars[0] == ' ' ? 1 : 0;
            for (size_t i = first; i < chars.size() && !subset.empty(); ++i)
                subset = nfa.step(subset, chars[i], work);
            if (subset.empty()) continue;
            Key next{false, std::move(subset)};
            auto it = indices.find(next);
            int dest;
            if (it == indices.end()) {
                if (keys.size() == kMaxTokenStates) fail("token state limit exceeded");
                dest = static_cast<int>(keys.size());
                indices.emplace(next, dest); keys.push_back(std::move(next)); states_.emplace_back();
            } else dest = it->second;
            if (++arcs > kMaxTokenArcs) fail("token arc limit exceeded");
            states_[q].edges.push_back({static_cast<int>(token), dest});
        }
    }
}

std::shared_ptr<const TokenGrammar> TokenGrammar::unconstrained(
        int token_count, const std::vector<int>& controls) {
    if (token_count <= 0 || static_cast<size_t>(token_count) > kMaxPieces)
        fail("unsupported vocabulary size");
    std::shared_ptr<TokenGrammar> grammar(new TokenGrammar());
    grammar->tokens_ = token_count;
    grammar->states_.emplace_back();
    auto& state = grammar->states_.back();
    state.accepting = true;
    for (int token = 0; token < token_count; ++token)
        if (std::find(controls.begin(), controls.end(), token) == controls.end())
            state.edges.push_back({token, 0});
    return grammar;
}

int TokenGrammar::transition(int state, int token) const {
    const auto& edges = arcs(state);
    auto it = std::lower_bound(edges.begin(), edges.end(), token,
                              [](const Edge& edge, int value) { return edge.token < value; });
    return it != edges.end() && it->token == token ? it->next : -1;
}
} // namespace pk
