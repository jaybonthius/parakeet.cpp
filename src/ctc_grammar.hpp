#pragma once

#include "grammar.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pk {

// Previous-raw-label CTC topology and best-alignment search over a text grammar.
class CtcGrammar {
public:
    static constexpr size_t kMaxRuleBytes = TokenGrammar::kMaxRuleBytes;
    CtcGrammar(const std::string& rules, const std::vector<std::string>& pieces,
               int blank_id);
    CtcGrammar(std::shared_ptr<const TokenGrammar> grammar, int blank_id);

    struct Result {
        bool matched = false;
        std::vector<int32_t> tokens;
        double score = 0;
    };
    Result decode(const std::vector<float>& log_probs, int frames, int classes) const;

private:
    struct CtcState {
        int grammar;
        int previous;
        std::vector<TokenGrammar::Edge> edges;
    };
    std::shared_ptr<const TokenGrammar> grammar_;
    int blank_;
    int classes_;
    std::vector<CtcState> ctc_states_;
};

} // namespace pk
