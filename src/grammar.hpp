#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace pk {

// Immutable text-token DFA for the readable regular grammar subset. State IDs
// belong to this grammar; restart each rendering unit at start() (one space trim).
class TokenGrammar {
public:
    static constexpr size_t kMaxRuleBytes = 65536;
    TokenGrammar(const std::string& rules, const std::vector<std::string>& pieces);

    struct Edge { int token; int next; };
    int start() const { return 0; }
    bool accepting(int state) const { return states_.at(state).accepting; }
    const std::vector<Edge>& arcs(int state) const { return states_.at(state).edges; }
    // Returns -1 for an illegal text token. Blank, EOU/EOB and other special
    // tokens are decoder controls, never text arcs; the decoder handles them.
    int transition(int state, int token) const;
    size_t state_count() const { return states_.size(); }
    int token_count() const { return tokens_; }

private:
    struct State {
        bool accepting = false;
        std::vector<Edge> edges;
    };
    int tokens_;
    std::vector<State> states_;
};

} // namespace pk
