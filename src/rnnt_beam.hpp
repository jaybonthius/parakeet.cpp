#pragma once
#include "grammar.hpp"
#include "joint.hpp"
#include "prediction.hpp"
#include <cstdint>
#include <memory>
#include <vector>

namespace pk {

struct RnntBeamToken {
    int32_t id;
    int64_t frame;
};

struct RnntAcceptedUnit {
    std::vector<RnntBeamToken> tokens;
    int endpoint_id;                 // EOU or EOB; never a text token
    int64_t frame;
};

// Offline controls preserve the request grammar; streaming controls close and
// restart a rendering unit immediately, but reset prediction only at chunk end.
enum class RnntEndpointPolicy { preserve, restart };

namespace detail {
struct RnntSearchLimits {
    size_t max_expansions = 1000000;
    size_t max_work = 100000000;
    size_t max_history = 65536;       // uncommitted text tokens + endpoints per path
};
struct RnntScoredTransition {
    int64_t frame;
    std::vector<int32_t> prefix;     // prediction history since its last reset
    std::vector<float> logits;
};
struct RnntBeamTestAccess;
} // namespace detail

struct RnntPredictionCache {
    std::vector<float> g;
    std::shared_ptr<const PredState> next;
};

struct RnntBeamHypothesis {
    double score = 0;
    int grammar_state = 0;
    std::vector<RnntBeamToken> tokens; // current unit only; controls excluded
    std::vector<RnntAcceptedUnit> units; // complete, not yet committed units
    std::shared_ptr<const PredState> predictor;
    int last_token = -1;             // SOS when -1
    std::shared_ptr<const RnntPredictionCache> cache;
    bool reset_at_chunk_end = false;
};

// Frame-synchronous bounded Viterbi beam: keeps alignment paths, not summed
// prefix probabilities; duplicate text prefixes can consume beam slots.
// Hard eligibility precedes beam pruning; probabilities
// use the entire model vocabulary. NO_MATCH means no retained accepting path,
// not acoustic rejection. Every frame advance pays its actual blank score,
// including after max_symbols emissions (no artificial time advance).
//
// One caller per frontier. The compiled grammar is retained. Model objects and
// backend/pool routing are borrowed only for advance(). Exceptions after an
// advance starts require reset() rather than replaying the same encoder frames.
class RnntGrammarBeam {
public:
    RnntGrammarBeam(std::shared_ptr<const TokenGrammar> grammar, int blank_id,
                    int beam_size = 4, int max_symbols = 10,
                    int eou_id = -1, int eob_id = -1,
                    RnntEndpointPolicy endpoints = RnntEndpointPolicy::preserve,
                    detail::RnntSearchLimits limits = {});

    // Row-major [frames, enc_hidden]. Partitioning alone has no reset/commit
    // effect. Absolute token/event frames start at zero and survive retirement.
    void advance(const PredictionNet& pred, const Joint& joint,
                 const std::vector<float>& enc, int frames, int enc_hidden);
    void reset();
    int64_t frame() const { return frame_; }
    const std::vector<RnntBeamHypothesis>& hypotheses() const { return beam_; }
    bool accepting(const RnntBeamHypothesis& hyp) const;
    const RnntBeamHypothesis* best(bool accepting_only = false) const;

    // Call ONLY at a native encoder-chunk boundary, after ALL its frames. Does
    // nothing in offline/preserve mode. Grammar/suffixes are never reset here.
    void end_chunk();

    // Select the chunk-final winner's complete units, prune histories that
    // disagree (including event positions), and retire the committed prefix.
    // Compatible later units and provisional suffixes remain live. The caller
    // queues returned units until successful output allocation/publication.
    std::vector<RnntAcceptedUnit> commit_units(bool accepting_only = false);

private:
    friend struct detail::RnntBeamTestAccess;
    template<class Scorer> void advance_impl(int frames, Scorer&& score);
    std::shared_ptr<const TokenGrammar> grammar_;
    int blank_id_, beam_size_, max_symbols_, eou_id_, eob_id_;
    RnntEndpointPolicy endpoints_;
    detail::RnntSearchLimits limits_;
    int64_t frame_ = 0;
    std::vector<RnntBeamHypothesis> beam_;
};

namespace detail {
// Fixed prefix-dependent raw logits, driving the identical carried search
// kernel without inference. Not a public/customizable scoring framework.
void rnnt_advance_scored(RnntGrammarBeam& beam,
                         const std::vector<RnntScoredTransition>& transitions,
                         int frames);
} // namespace detail
} // namespace pk
