#include "rnnt_beam.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace pk {
namespace {

void charge(size_t& used, size_t count, size_t limit) {
    if (used > limit || count > limit - used)
        throw std::runtime_error("RNNT grammar search resource limit");
    used += count;
}

size_t history_size(const RnntBeamHypothesis& hyp) {
    size_t count = hyp.tokens.size();
    for (const auto& unit : hyp.units) count += unit.tokens.size() + 1;
    return count;
}

bool same_unit(const RnntAcceptedUnit& a, const RnntAcceptedUnit& b) {
    if (a.endpoint_id != b.endpoint_id || a.frame != b.frame ||
        a.tokens.size() != b.tokens.size()) return false;
    for (size_t i = 0; i < a.tokens.size(); ++i)
        if (a.tokens[i].id != b.tokens[i].id || a.tokens[i].frame != b.tokens[i].frame)
            return false;
    return true;
}

void prune(std::vector<RnntBeamHypothesis>& paths, int width) {
    // Stable order gives deterministic ties without merging predictor histories.
    std::stable_sort(paths.begin(), paths.end(), [](const auto& a, const auto& b) {
        return a.score > b.score;
    });
    if (paths.size() > (size_t)width) paths.resize((size_t)width);
}

std::vector<double> log_probs(const std::vector<float>& logits, int blank) {
    if (logits.size() != (size_t)blank + 1)
        throw std::invalid_argument("RNNT grammar logits shape mismatch");
    for (float value : logits)
        if (!std::isfinite(value)) throw std::runtime_error("RNNT grammar nonfinite logits");
    const double top = *std::max_element(logits.begin(), logits.end());
    double sum = 0;
    for (float value : logits) sum += std::exp((double)value - top);
    const double norm = std::log(sum);
    std::vector<double> result;
    result.reserve(logits.size());
    for (float value : logits) result.push_back(((double)value - top) - norm);
    return result;
}

} // namespace

RnntGrammarBeam::RnntGrammarBeam(std::shared_ptr<const TokenGrammar> grammar,
                                 int blank_id, int beam_size, int max_symbols,
                                 int eou_id, int eob_id, RnntEndpointPolicy endpoints,
                                 detail::RnntSearchLimits limits)
    : grammar_(std::move(grammar)), blank_id_(blank_id), beam_size_(beam_size),
      max_symbols_(max_symbols), eou_id_(eou_id), eob_id_(eob_id),
      endpoints_(endpoints), limits_(limits) {
    if (!grammar_ || blank_id_ != grammar_->token_count() || blank_id_ <= 0)
        throw std::invalid_argument("RNNT grammar vocabulary mismatch");
    if (beam_size_ < 1 || beam_size_ > 64 || max_symbols_ < 1 || max_symbols_ > 128)
        throw std::invalid_argument("RNNT grammar beam/symbol bounds");
    if (eou_id_ < -1 || eou_id_ >= blank_id_ || eob_id_ < -1 || eob_id_ >= blank_id_ ||
        (eou_id_ >= 0 && eou_id_ == eob_id_))
        throw std::invalid_argument("RNNT grammar endpoint IDs");
    // A control ID must not also be a legal text arc in this grammar.
    for (size_t s = 0; s < grammar_->state_count(); ++s)
        for (const auto& arc : grammar_->arcs((int)s))
            if (arc.token == eou_id_ || arc.token == eob_id_)
                throw std::invalid_argument("RNNT grammar endpoint is a text token");
    reset();
}

void RnntGrammarBeam::reset() {
    RnntBeamHypothesis initial;
    initial.grammar_state = grammar_->start();
    std::vector<RnntBeamHypothesis> fresh;
    fresh.push_back(std::move(initial));
    beam_.swap(fresh);
    frame_ = 0;
}

bool RnntGrammarBeam::accepting(const RnntBeamHypothesis& hyp) const {
    return grammar_->accepting(hyp.grammar_state);
}

const RnntBeamHypothesis* RnntGrammarBeam::best(bool accepting_only) const {
    for (const auto& hyp : beam_)
        if (!accepting_only || accepting(hyp)) return &hyp;
    return nullptr;
}

template<class Scorer>
void RnntGrammarBeam::advance_impl(int frames, Scorer&& score) {
    if (frames < 0 || frame_ > std::numeric_limits<int64_t>::max() - frames)
        throw std::invalid_argument("RNNT grammar frame bounds");
    size_t expansions = 0, work = 0;
    struct Candidate { size_t source; int token; int next; double score; };
    for (int t = 0; t < frames; ++t) {
        for (const auto& hyp : beam_) charge(work, history_size(hyp), limits_.max_work);
        auto active = beam_;
        std::vector<RnntBeamHypothesis> completed;
        for (int symbols = 0; symbols <= max_symbols_ && !active.empty(); ++symbols) {
            std::vector<Candidate> candidates;
            for (size_t i = 0; i < active.size(); ++i) {
                auto& hyp = active[i];
                charge(expansions, 1, limits_.max_expansions);
                charge(work, (size_t)blank_id_ + 1, limits_.max_work);
                const auto lp = log_probs(score(hyp, t, frame_), blank_id_);
                charge(work, history_size(hyp), limits_.max_work);
                auto blank = hyp;
                blank.score += lp[blank_id_];
                if (!std::isfinite(blank.score))
                    throw std::runtime_error("RNNT grammar nonfinite score");
                completed.push_back(std::move(blank));
                if (symbols == max_symbols_) continue;

                const auto& arcs = grammar_->arcs(hyp.grammar_state);
                charge(work, arcs.size() + 2, limits_.max_work);
                std::vector<Candidate> eligible;
                eligible.reserve(arcs.size() + 2);
                for (const auto& arc : arcs)
                    eligible.push_back({i, arc.token, arc.next, hyp.score + lp[arc.token]});
                if (accepting(hyp)) {
                    for (int control : {eou_id_, eob_id_})
                        if (control >= 0)
                            eligible.push_back({i, control, hyp.grammar_state,
                                                hyp.score + lp[control]});
                }
                const size_t keep = std::min(eligible.size(), (size_t)beam_size_);
                std::partial_sort(eligible.begin(), eligible.begin() + keep, eligible.end(),
                    [](const auto& a, const auto& b) {
                        return a.score != b.score ? a.score > b.score : a.token < b.token;
                    });
                candidates.insert(candidates.end(), eligible.begin(), eligible.begin() + keep);
            }
            prune(completed, beam_size_);
            std::stable_sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                return a.score > b.score;
            });
            if (candidates.size() > (size_t)beam_size_) candidates.resize((size_t)beam_size_);
            // Future log probabilities cannot improve an active path's score.
            if (completed.size() == (size_t)beam_size_ && !candidates.empty() &&
                candidates.front().score <= completed.back().score) break;
            std::vector<RnntBeamHypothesis> next;
            next.reserve(candidates.size());
            for (const auto& c : candidates) {
                if (!std::isfinite(c.score)) throw std::runtime_error("RNNT grammar nonfinite score");
                charge(work, history_size(active[c.source]), limits_.max_work);
                auto child = active[c.source];
                if (history_size(child) >= limits_.max_history)
                    throw std::runtime_error("RNNT grammar history resource limit");
                child.score = c.score;
                child.predictor = child.cache->next;
                child.last_token = c.token;
                child.cache.reset();
                if (c.token == eou_id_ || c.token == eob_id_) {
                    if (endpoints_ == RnntEndpointPolicy::restart) {
                        child.units.push_back({std::move(child.tokens), c.token, frame_});
                        child.tokens.clear();
                        child.grammar_state = grammar_->start();
                        child.reset_at_chunk_end = true;
                    }
                } else {
                    child.tokens.push_back({c.token, frame_});
                    child.grammar_state = c.next;
                }
                next.push_back(std::move(child));
            }
            active = std::move(next);
        }
        beam_ = std::move(completed);
        ++frame_;
    }
}

void RnntGrammarBeam::advance(const PredictionNet& pred, const Joint& joint,
                               const std::vector<float>& enc, int frames, int enc_hidden) {
    if (frames < 0 || enc_hidden <= 0 || enc.size() != (size_t)frames * enc_hidden)
        throw std::invalid_argument("RNNT grammar encoder shape mismatch");
    if (joint.num_durations() != 0 || joint.vocab_size() != blank_id_ ||
        joint.V_plus() != blank_id_ + 1)
        throw std::invalid_argument("RNNT grammar incompatible joint head");
    if (frame_ > std::numeric_limits<int64_t>::max() - frames)
        throw std::invalid_argument("RNNT grammar frame bounds");
    if (!frames) return;
    std::vector<float> projected;
    joint.precompute_enc_proj(enc, frames, enc_hidden, projected);
    std::vector<float> logits;
    advance_impl(frames, [&](RnntBeamHypothesis& hyp, int local, int64_t) -> const std::vector<float>& {
        if (!hyp.cache) {
            auto cache = std::make_shared<RnntPredictionCache>();
            auto next = std::make_shared<PredState>();
            const PredState in = hyp.predictor ? PredState{} : pred.zero_state();
            pred.step(hyp.last_token < 0 ? blank_id_ : hyp.last_token, hyp.last_token < 0,
                      hyp.predictor ? *hyp.predictor : in, cache->g, *next);
            cache->next = std::move(next);
            hyp.cache = std::move(cache);
        }
        joint.step_logits(projected.data() + (size_t)local * joint.joint_hidden(),
                          hyp.cache->g.data(), (int)hyp.cache->g.size(), logits);
        return logits;
    });
}

void RnntGrammarBeam::end_chunk() {
    if (endpoints_ != RnntEndpointPolicy::restart) return;
    for (auto& hyp : beam_) {
        if (!hyp.reset_at_chunk_end) continue;
        hyp.predictor.reset();
        hyp.last_token = -1;
        hyp.cache.reset();
        hyp.reset_at_chunk_end = false;
    }
}

std::vector<RnntAcceptedUnit> RnntGrammarBeam::commit_units(bool accepting_only) {
    if (endpoints_ != RnntEndpointPolicy::restart) return {};
    const auto* selected = best(accepting_only);
    if (!selected || selected->units.empty()) return {};
    auto units = selected->units;
    std::vector<RnntBeamHypothesis> compatible;
    for (const auto& hyp : beam_) {
        if (hyp.units.size() < units.size()) continue;
        bool match = true;
        for (size_t i = 0; i < units.size(); ++i)
            if (!same_unit(hyp.units[i], units[i])) { match = false; break; }
        if (!match) continue;
        auto kept = hyp;
        kept.units.erase(kept.units.begin(), kept.units.begin() + units.size());
        compatible.push_back(std::move(kept));
    }
    beam_.swap(compatible);
    return units;
}

namespace detail {
struct RnntBeamTestAccess {
    static void advance(RnntGrammarBeam& beam,
                        const std::vector<RnntScoredTransition>& rows, int frames) {
        beam.advance_impl(frames, [&](RnntBeamHypothesis& hyp, int, int64_t frame) -> const std::vector<float>& {
            if (!hyp.cache) {
                auto cache = std::make_shared<RnntPredictionCache>();
                auto next = std::make_shared<PredState>();
                next->h = hyp.predictor ? hyp.predictor->h : std::vector<std::vector<float>>{{}};
                if (hyp.last_token >= 0) next->h[0].push_back((float)hyp.last_token);
                cache->g = next->h[0];
                cache->next = std::move(next);
                hyp.cache = std::move(cache);
            }
            std::vector<int32_t> prefix;
            for (float value : hyp.cache->g) prefix.push_back((int32_t)value);
            for (const auto& row : rows)
                if (row.frame == frame && row.prefix == prefix) return row.logits;
            throw std::runtime_error("RNNT grammar missing scored transition");
        });
    }
};
void rnnt_advance_scored(RnntGrammarBeam& beam,
                         const std::vector<RnntScoredTransition>& transitions, int frames) {
    RnntBeamTestAccess::advance(beam, transitions, frames);
}
} // namespace detail
} // namespace pk
