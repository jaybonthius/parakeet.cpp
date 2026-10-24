#include "ctc_grammar.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace pk {
namespace {
constexpr size_t kMaxTransitions = 20000000;
constexpr size_t kMaxTraceCells = 16777216;
constexpr size_t kMaxDecodeWork = 100000000;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error("CTC grammar: " + message);
}
} // namespace

CtcGrammar::CtcGrammar(const std::string& rules, const std::vector<std::string>& pieces, int blank_id)
    : CtcGrammar(std::make_shared<const TokenGrammar>(rules, pieces), blank_id) {}

CtcGrammar::CtcGrammar(std::shared_ptr<const TokenGrammar> grammar, int blank_id)
    : grammar_(std::move(grammar)), blank_(blank_id), classes_(0) {
    if (!grammar_ || blank_id != grammar_->token_count()) fail("unsupported CTC vocabulary/blank layout");
    classes_ = grammar_->token_count() + 1;
    // Only incoming labels (plus blank) can occur as the previous CTC label at a DFA state.
    std::map<std::pair<int, int>, int> products;
    for (size_t q = 0; q < grammar_->state_count(); ++q) {
        products[{static_cast<int>(q), blank_}] = static_cast<int>(ctc_states_.size());
        ctc_states_.push_back({static_cast<int>(q), blank_, {}});
    }
    for (size_t q = 0; q < grammar_->state_count(); ++q) for (auto arc : grammar_->arcs(static_cast<int>(q))) {
        auto key = std::make_pair(arc.next, arc.token);
        if (!products.count(key)) {
            products[key] = static_cast<int>(ctc_states_.size());
            ctc_states_.push_back({arc.next, arc.token, {}});
        }
    }
    size_t transitions = 0;
    for (size_t i = 0; i < ctc_states_.size(); ++i) {
        auto& state = ctc_states_[i];
        state.edges.push_back({blank_, products.at({state.grammar, blank_})});
        if (state.previous != blank_) state.edges.push_back({state.previous, static_cast<int>(i)});
        for (auto arc : grammar_->arcs(state.grammar)) {
            if (++transitions > kMaxTransitions) fail("CTC transition limit exceeded");
            if (arc.token != state.previous)
                state.edges.push_back({arc.token, products.at({arc.next, arc.token})});
        }
    }
}

CtcGrammar::Result CtcGrammar::decode(const std::vector<float>& log_probs, int frames, int classes) const {
    if (frames < 0 || classes != classes_ ||
        static_cast<size_t>(frames) > std::numeric_limits<size_t>::max() / static_cast<size_t>(classes) ||
        log_probs.size() != static_cast<size_t>(frames) * classes) fail("invalid log-probability dimensions");
    const size_t count = ctc_states_.size();
    if (static_cast<size_t>(frames) + 1 > kMaxTraceCells / count) fail("traceback cell limit exceeded");
    size_t transitions = 0;
    for (const auto& state : ctc_states_) transitions += state.edges.size();
    if (frames && transitions > kMaxDecodeWork / static_cast<size_t>(frames)) fail("decode work limit exceeded");
    for (float p : log_probs) if (std::isnan(p) || p > 0) fail("invalid log-probability value");
    const double impossible = -std::numeric_limits<double>::infinity();
    std::vector<double> current(count, impossible), next(count, impossible);
    std::vector<int32_t> trace((static_cast<size_t>(frames) + 1) * count, -1);
    current[0] = 0;
    for (int t = 0; t < frames; ++t) {
        std::fill(next.begin(), next.end(), impossible);
        for (size_t i = 0; i < count; ++i) {
            if (current[i] == impossible) continue;
            for (auto edge : ctc_states_[i].edges) {
                double score = current[i] + log_probs[static_cast<size_t>(t) * classes + edge.token];
                // Stable iteration order breaks exact ties; do not renormalize legal labels.
                if (score > next[edge.next]) {
                    next[edge.next] = score;
                    trace[(static_cast<size_t>(t) + 1) * count + edge.next] = static_cast<int32_t>(i);
                }
            }
        }
        current.swap(next);
    }
    int best = -1;
    double score = impossible;
    for (size_t i = 0; i < count; ++i)
        if (grammar_->accepting(ctc_states_[i].grammar) && current[i] > score) {
            score = current[i]; best = static_cast<int>(i);
        }
    Result result;
    if (best < 0) return result;
    result.matched = true; result.score = score;
    std::vector<int> alignment(static_cast<size_t>(frames));
    for (int t = frames; t > 0; --t) {
        alignment[t - 1] = ctc_states_[best].previous;
        best = trace[static_cast<size_t>(t) * count + best];
    }
    int previous = blank_;
    for (int token : alignment) {
        if (token != blank_ && token != previous) result.tokens.push_back(token);
        previous = token;
    }
    return result;
}
} // namespace pk
