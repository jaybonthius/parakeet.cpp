#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
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

    // One accepting state with an arc for every token except `controls`: the
    // unconstrained reference search run beside a grammar search.
    static std::shared_ptr<const TokenGrammar> unconstrained(int token_count,
                                                             const std::vector<int>& controls);

private:
    TokenGrammar() = default;
    struct State {
        bool accepting = false;
        std::vector<Edge> edges;
    };
    int tokens_ = 0;
    std::vector<State> states_;
};

// Scores reported with a grammar result. Every score is a raw sum of natural-log
// model probabilities over the same `frames` encoder frames (no length
// normalization), so scores of one result compare directly. They are search
// scores, not calibrated confidence.
struct GrammarAlternative {
    std::string text;
    double score = 0;                // log probability of its accepted path
};
struct GrammarScores {
    double score = 0;                // log probability of the accepted path
    double free_score = 0;           // same search without the grammar
    int64_t frames = 0;
    // Distinct next-best accepted texts the search retained, best first.
    std::vector<GrammarAlternative> alternatives;
};

} // namespace pk
