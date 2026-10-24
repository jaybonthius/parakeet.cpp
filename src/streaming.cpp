#include "streaming.hpp"
#include "tokenizer.hpp"
#include "mel.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace pk {
namespace {
void apply_stream_prompt(const PromptKernel& prompt, int index,
                         std::vector<float>& enc, int frames, int hidden) {
    if (prompt.present()) {
        std::vector<float> chunk_cf((size_t)hidden * frames);  // [d_model, valid]
        for (int t = 0; t < frames; ++t)
            for (int c = 0; c < hidden; ++c)
                chunk_cf[(size_t)c * frames + t] = enc[(size_t)t * hidden + c];
        std::vector<float> projected;
        prompt.apply(chunk_cf, hidden, frames, index, projected);  // [d_model, valid]
        for (int t = 0; t < frames; ++t)
            for (int c = 0; c < hidden; ++c)
                enc[(size_t)t * hidden + c] = projected[(size_t)c * frames + t];
    }
}
} // namespace

StreamingSession::StreamingSession(const ModelLoader& ml, const std::string& target_lang)
    : ml_(ml), enc_(ml), pred_(ml), joint_(ml), prompt_(ml) {
    const ParakeetConfig& cfg = ml.config();
    d_model_  = (int)cfg.d_model;
    blank_id_ = (int)cfg.blank_id;

    // Resolve the language prompt index for multilingual (nemotron) models. The
    // one-hot is constant over time (one language per utterance), so the prompt
    // projection is applied per chunk in feed_mel_chunk using this fixed index.
    // Non-prompt models leave prompt_index_ = -1 and skip prompt_.apply().
    if (cfg.prompt.present) {
        // Empty target_lang -> the model default; an unknown locale THROWS
        // std::runtime_error (same message as Model::resolve_prompt_index), so a
        // typo (e.g. --lang xx) fails loudly instead of silently mis-transcribing.
        // Matches the offline path and the parakeet_capi_stream_begin_lang
        // contract (NULL + ctx last_error on an unknown locale).
        prompt_index_ = cfg.prompt.resolve_index_or_throw(target_lang);
    }
    // Greedy max symbols per frame, from model metadata (NeMo default 10);
    // matches the offline pk::transcribe path in model.cpp.
    max_symbols_ = (int)cfg.max_symbols;
    assert(joint_.num_durations() == 0 && "StreamingSession is RNN-T only (no TDT durations)");

    // Resolve the <EOU>/<EOB> special token ids from the tokenizer pieces (do
    // NOT hardcode 1024/1025; read them from the loaded vocab).
    const auto& pieces = cfg.tokenizer_pieces;
    for (int i = 0; i < (int)pieces.size(); ++i) {
        if (pieces[i] == "<EOU>") eou_id_ = i;
        else if (pieces[i] == "<EOB>") eob_id_ = i;
    }

    // Per-encoder-frame stride in seconds: each encoder output frame spans
    // hop_length * subsampling_factor input samples (the subsampling downsamples
    // the mel-frame time axis by subsampling_factor). time = frame * stride.
    const double hop = (double)cfg.hop_length;
    const double sub = (double)(cfg.subsampling_factor ? cfg.subsampling_factor : 1);
    const double sr  = (double)(cfg.sample_rate ? cfg.sample_rate : 16000);
    frame_sec_   = (hop * sub) / sr;
    frame_sec_f_ = (float)frame_sec_;

    reset();
}

void StreamingSession::reset() {
    enc_.reset();
    state_ = rnnt_decode_init(pred_);
    // eou_id_/eob_id_/frame_sec_ are resolved once in the constructor.
    enc_frame_ = 0;
    non_special_.clear();
    text_.clear();
    text_taken_ = 0;
    last_chunk_had_eou_ = false;
    events_.clear();
    word_tokens_.clear();
    words_.clear();
    words_finalized_ = 0;
    words_taken_ = 0;
}

void StreamingSession::process_emitted(const std::vector<int32_t>& emitted) {
    last_chunk_had_eou_ = false;
    bool text_changed = false;
    for (int32_t tok : emitted) {
        if (tok == eou_id_ || tok == eob_id_) {
            // EOU/EOB: surface as an event, do NOT add to the text.
            EouEvent ev;
            ev.token = tok;
            ev.is_eob = (tok == eob_id_);
            // The frame index is recorded into events_ by feed/finalize (which
            // knows the per-token frame); we patch it there. Here we only know
            // the token, so push a placeholder the caller will overwrite.
            ev.encoder_frame = enc_frame_;  // refined by caller via per-token frame
            ev.time_sec = ev.encoder_frame * frame_sec_;
            events_.push_back(ev);
            last_chunk_had_eou_ = true;
        } else {
            non_special_.push_back(tok);
            text_changed = true;
        }
    }
    if (text_changed) {
        text_ = detokenize(ml_.config().tokenizer_pieces, non_special_);
    }
}

std::vector<int32_t> StreamingSession::feed_mel_chunk(const std::vector<float>& mel_chunk,
                                                      int n_frames, bool is_last) {
    // 1. Encoder step: the chunk's valid encoder frames, row-major [valid, d_model]
    //    (d_model fastest) — exactly the orientation rnnt_decode_frames expects.
    int n_valid = 0;
    std::vector<float> enc_frames = enc_.step(mel_chunk, n_frames, is_last, n_valid);

    if (n_valid <= 0) {
        last_chunk_had_eou_ = false;
        return {};
    }

    // 1b. Prompt conditioning (nemotron multilingual): project the chunk's
    //     encoder frames through prompt_kernel for the resolved language before
    //     the RNN-T decode. The one-hot is constant over time, so applying it
    //     per chunk is exact (== the offline forward's single application).
    //     prompt_.apply() wants channels-first [d_model, valid]; enc_.step gives
    //     time-major [valid, d_model], so transpose in, apply, transpose back.
    //     No-op for non-prompt models (prompt_.present()==false): enc_frames is
    //     left byte-identical.
    apply_stream_prompt(prompt_, prompt_index_, enc_frames, n_valid, d_model_);

    // 2. RNN-T greedy over the new encoder frames, carrying the decoder state
    //    across chunks (do NOT reset). Appends to state_.hyp and returns the ids
    //    emitted in this chunk, with their LOCAL frame index in [0, n_valid) and
    //    per-token TokenInfo (LOCAL frame, max_prob conf, span==1).
    const int base_frame = enc_frame_;
    std::vector<int32_t> local_frames;
    std::vector<TokenInfo> chunk_tokens;
    std::vector<int32_t> emitted =
        rnnt_decode_frames(pred_, joint_, enc_frames, n_valid, d_model_,
                           state_, blank_id_, max_symbols_, &local_frames,
                           &chunk_tokens);
    enc_frame_ += n_valid;

    // 3. Update text + EOU events; refine each new event's absolute frame index
    //    from the per-token local frame the decoder reported.
    const size_t prev_events = events_.size();
    process_emitted(emitted);
    // Re-walk emitted to assign the correct absolute frame to each new event,
    // and accumulate NON-special tokens (absolute frame) for word grouping.
    size_t evi = prev_events;
    for (size_t i = 0; i < emitted.size(); ++i) {
        if (emitted[i] == eou_id_ || emitted[i] == eob_id_) {
            const int abs_frame = base_frame + (int)local_frames[i];
            events_[evi].encoder_frame = abs_frame;
            events_[evi].time_sec = abs_frame * frame_sec_;
            ++evi;
        } else {
            TokenInfo ti = chunk_tokens[i];
            ti.frame += base_frame;  // local -> absolute encoder frame
            word_tokens_.push_back(ti);
        }
    }
    // Regroup the accumulated tokens; words before the last (still-open) one are
    // final and become available to drain_words().
    regroup_words(/*flush_all=*/false);

    // 4. End-of-utterance reset. The realtime EOU model is trained to emit <EOU>
    //    (end of utterance) / <EOB> (backchannel) and have the decoder START THE
    //    NEXT UTTERANCE FROM A FRESH STATE — exactly what NeMo's reference
    //    streaming driver does (examples/voice_agent .../nemo/streaming_asr.py
    //    NemoStreamingASRService.transcribe -> reset_state() when <EOU>/<EOB>
    //    appears in the chunk text). Without this the prediction net stays
    //    conditioned on the just-emitted <EOU> and the joint scores blank on every
    //    subsequent frame, so the stream goes silent after the first utterance
    //    (issue #13). We reset the carried RNN-T decoder state — LSTM h/c to zero
    //    and last_token back to SOS — for the next chunk. We deliberately reset
    //    ONLY the decoder, not the StreamingEncoder cache: NeMo's reset_state also
    //    drops the encoder cache, but that was verified to be a no-op for the
    //    decoded tokens (decoder-only reset == NeMo's full reset_state byte-for-
    //    byte on multi-utterance clips), so the validated streaming-encoder path is
    //    left untouched. enc_frame_ keeps running so <EOU> timestamps stay absolute
    //    in the clip, and state_.hyp keeps the full token record across utterances.
    if (last_chunk_had_eou_) {
        state_.state      = pred_.zero_state();
        state_.last_token = -1;     // SOS sentinel (nothing emitted yet)
        state_.have_token = false;
    }
    return emitted;
}

std::string StreamingSession::finalize() {
    // The end-of-stream tail is flushed by the caller feeding the final buffered
    // mel chunk with is_last=true (which keeps the streaming tail frames). At the
    // session level there is no further audio to process here, so finalize just
    // returns whatever newly-finalized text remains since the last take. NeMo's
    // cache-aware streaming behaves identically: the final chunk's incomplete
    // right context means a trailing <EOU> is NOT recovered, so we never
    // fabricate one — finalize emits only the carried-state tokens already
    // decoded.
    //
    // Word side: the end-of-stream has no further word-start markers, so the
    // trailing open word is now final too — regroup with flush_all so it becomes
    // available to drain_words().
    regroup_words(/*flush_all=*/true);
    return take_new_text();
}

void StreamingSession::regroup_words(bool flush_all) {
    // Re-run the validated offline grouping over the whole accumulated
    // non-special token sequence (it does the punctuation lookahead / refinement
    // exactly like the offline transcribe_with_timestamps path). The last word is
    // still "open" (its text/end can change when more tokens arrive), so only
    // words BEFORE it are considered final mid-stream; flush_all makes every word
    // final at end-of-stream.
    words_ = group_words(word_tokens_, ml_.config().tokenizer_pieces, frame_sec_f_);
    if (words_.empty()) {
        words_finalized_ = 0;
    } else {
        words_finalized_ = flush_all ? words_.size() : (words_.size() - 1);
    }
    // Never "un-finalize" a word we've already handed out.
    if (words_finalized_ < words_taken_) words_finalized_ = words_taken_;
}

std::vector<Word> StreamingSession::drain_words() {
    std::vector<Word> out;
    for (size_t i = words_taken_; i < words_finalized_ && i < words_.size(); ++i)
        out.push_back(words_[i]);
    words_taken_ = words_finalized_;
    return out;
}

std::string StreamingSession::take_new_text() {
    if (text_taken_ >= text_.size()) return std::string();
    std::string delta = text_.substr(text_taken_);
    text_taken_ = text_.size();
    return delta;
}

std::vector<EouEvent> StreamingSession::drain_events() {
    std::vector<EouEvent> out;
    out.swap(events_);
    return out;
}

namespace detail {

NativeMelChunks::NativeMelChunks(int n_mels, int first, int main, int overlap)
    : n_mels_(n_mels), first_size_(first), main_size_(main), overlap_(overlap) {
    if (n_mels <= 0 || first <= 0 || main <= 0 || overlap < 0 ||
        overlap > std::numeric_limits<int>::max() - std::max(first, main))
        throw std::invalid_argument("invalid native streaming schedule");
}

void NativeMelChunks::reset() noexcept {
    total_ = origin_ = next_ = 0;
    first_ = true;
    mel_.clear();
}

void NativeMelChunks::append(const std::vector<float>& frames, int count) {
    if (n_mels_ <= 0 || count < 0 || frames.size() != (size_t)n_mels_ * count)
        throw std::invalid_argument("invalid streaming mel frames");
    if (count == 0) return;
    if (total_ > std::numeric_limits<int64_t>::max() - count)
        throw std::runtime_error("streaming mel timeline limit");
    // Dropping only unreachable history avoids O(stream length) copies and
    // allocator fragmentation while preserving every native overlap window.
    const int64_t keep = std::min(total_, std::max(origin_, next_ - overlap_));
    const size_t old_len = (size_t)(total_ - origin_);
    const size_t drop = (size_t)(keep - origin_);
    const size_t kept = old_len - drop;
    const size_t len = kept + (size_t)count;
    if (len > std::vector<float>().max_size() / (size_t)n_mels_)
        throw std::runtime_error("streaming mel buffer limit");
    std::vector<float> out((size_t)n_mels_ * len);
    for (int m = 0; m < n_mels_; ++m) {
        std::copy_n(mel_.begin() + (size_t)m * old_len + drop, kept,
                    out.begin() + (size_t)m * len);
        std::copy_n(frames.begin() + (size_t)m * count, count,
                    out.begin() + (size_t)m * len + kept);
    }
    mel_.swap(out);
    total_ += count;
    origin_ = keep;
}

void NativeMelChunks::feed(bool flush,
        const std::function<void(const std::vector<float>&, int, bool)>& chunk) {
    if (n_mels_ <= 0) throw std::logic_error("uninitialized streaming schedule");
    while (next_ < total_) {
        const int size = first_ ? first_size_ : main_size_;
        const int64_t hi = next_ + std::min<int64_t>(size, total_ - next_);
        const bool reaches_end = hi == total_;
        if (!flush && reaches_end) break;
        const int64_t lo = first_ ? next_ : std::max<int64_t>(0, next_ - overlap_);
        const int len = (int)(hi - lo);
        const size_t local_len = (size_t)(total_ - origin_);
        std::vector<float> window((size_t)n_mels_ * len);
        for (int m = 0; m < n_mels_; ++m)
            std::copy_n(mel_.begin() + (size_t)m * local_len + (size_t)(lo - origin_),
                        len, window.begin() + (size_t)m * len);
        chunk(window, len, flush && reaches_end);
        // On flush a short final chunk consumes exactly the remaining frames.
        next_ = hi;
        first_ = false;
    }
}

GrammarStreamState::GrammarStreamState(std::shared_ptr<const TokenGrammar> grammar,
        const std::vector<std::string>& pieces, int blank, int beam_size,
        int max_symbols, int eou, int eob, double frame_sec)
    : pieces_(pieces), beam_(std::move(grammar), blank, beam_size, max_symbols,
                            eou, eob, RnntEndpointPolicy::restart),
      eob_(eob), frame_sec_(frame_sec) {
    if (pieces.size() != (size_t)blank || !std::isfinite(frame_sec) || frame_sec <= 0)
        throw std::invalid_argument("invalid grammar stream vocabulary/timing");
}

std::string GrammarStreamState::render(const std::vector<RnntBeamToken>& tokens) const {
    std::vector<int32_t> ids;
    ids.reserve(tokens.size());
    for (const auto& token : tokens) ids.push_back(token.id);
    return detokenize(pieces_, ids);
}

void GrammarStreamState::commit(const std::vector<RnntAcceptedUnit>& accepted) {
    for (const auto& unit : accepted) {
        units_.push_back({render(unit.tokens), unit.endpoint_id, unit.endpoint_id == eob_,
                          unit.frame, unit.frame * frame_sec_});
        committed_ = true;
    }
}

void GrammarStreamState::end_chunk() {
    beam_.end_chunk();
    commit(beam_.commit_units());
    const auto* best = beam_.best();
    partial_ = best ? render(best->tokens) : std::string();
}

bool GrammarStreamState::finalize() {
    const auto& paths = beam_.hypotheses();
    const auto accepted = std::find_if(paths.begin(), paths.end(), [&](const auto& hyp) {
        const bool clean_restart = hyp.tokens.empty() && (committed_ || !hyp.units.empty());
        return beam_.accepting(hyp) || clean_restart;
    });
    if (accepted == paths.end()) return false;
    // Decoding is terminal: publish endpoints and tail from this same winner,
    // ranking clean restarts together with DFA-accepting suffixes.
    commit(accepted->units);
    if (!accepted->tokens.empty())
        units_.push_back({render(accepted->tokens), -1, false,
                          beam_.frame(), beam_.frame() * frame_sec_});
    partial_.clear();
    return true;
}

void GrammarStreamState::reset() {
    beam_.reset();
    units_.clear();
    partial_.clear();
    committed_ = false;
}

} // namespace detail

namespace {
const ModelLoader& validate_grammar_stream(const ModelLoader& ml) {
    const auto& c = ml.config();
    if (!c.streaming.present || !c.tdt_durations.empty() || !c.pred_hidden ||
        !c.pred_rnn_layers || !c.joint_hidden)
        throw std::invalid_argument("grammar streaming requires a cache-aware RNNT model");
    if (c.normalize == "per_feature")
        throw std::invalid_argument("grammar streaming requires frame-local mel normalization");
    if (!c.causal_downsampling || !c.conv_causal || c.att_context_style != "chunked_limited" ||
        !c.n_heads || !c.d_model || c.d_model % c.n_heads || !c.hop_length ||
        !c.sample_rate || !c.subsampling_factor)
        throw std::invalid_argument("invalid grammar streaming model configuration");
    return ml;
}
int control_id(const ModelLoader& ml, const char* name) {
    const auto& pieces = ml.config().tokenizer_pieces;
    const auto found = std::find(pieces.begin(), pieces.end(), name);
    return found == pieces.end() ? -1 : (int)(found - pieces.begin());
}
} // namespace

GrammarStreamingSession::GrammarStreamingSession(const ModelLoader& ml,
        std::shared_ptr<const TokenGrammar> grammar, int beam_size,
        const std::string& target_lang)
    : ml_(validate_grammar_stream(ml)), mel_(ml_), enc_(ml_), pred_(ml_), joint_(ml_),
      prompt_(ml_), prompt_index_(ml_.config().prompt.present
          ? ml_.config().prompt.resolve_index_or_throw(target_lang) : -1),
      chunks_(mel_.n_mels(), enc_.chunk_size_first(), enc_.chunk_size(), enc_.pre_encode_cache_size()),
      output_(std::move(grammar), ml_.config().tokenizer_pieces, (int)ml_.config().blank_id,
              beam_size, (int)ml_.config().max_symbols, control_id(ml_, "<EOU>"),
              control_id(ml_, "<EOB>"), (double)ml_.config().hop_length *
              ml_.config().subsampling_factor / ml_.config().sample_rate) {
    if (joint_.num_durations() != 0)
        throw std::invalid_argument("grammar streaming requires an RNNT joint");
}

void GrammarStreamingSession::require_active() const {
    if (!active()) throw std::logic_error("grammar stream requires reset");
}

void GrammarStreamingSession::feed_chunk(const std::vector<float>& mel, int frames, bool last) {
    int valid = 0;
    auto encoded = enc_.step(mel, frames, last, valid);
    if (valid > 0) {
        const int hidden = (int)ml_.config().d_model;
        apply_stream_prompt(prompt_, prompt_index_, encoded, valid, hidden);
        output_.beam().advance(pred_, joint_, encoded, valid, hidden);
    }
    output_.end_chunk();
}

void GrammarStreamingSession::feed(const float* pcm, int n_samples) {
    require_active();
    try {
        if (n_samples < 0 || (n_samples > 0 && !pcm))
            throw std::invalid_argument("invalid grammar stream PCM buffer");
        for (int i = 0; i < n_samples; ++i)
            if (!std::isfinite(pcm[i])) throw std::invalid_argument("nonfinite grammar stream PCM");
        if (n_samples > 0) {
            int count = 0;
            auto frames = mel_.feed(pcm, n_samples, count);
            chunks_.append(frames, count);
        }
        chunks_.feed(false, [this](const auto& mel, int frames, bool last) {
            feed_chunk(mel, frames, last);
        });
    } catch (...) { fail(); throw; }
}

bool GrammarStreamingSession::finalize() {
    require_active();
    try {
        int count = 0;
        auto tail = mel_.finalize(count);
        chunks_.append(tail, count);
        chunks_.feed(true, [this](const auto& mel, int frames, bool last) {
            feed_chunk(mel, frames, last);
        });
        const bool matched = output_.finalize();
        lifecycle_ = Lifecycle::finalized;
        return matched;
    } catch (...) { fail(); throw; }
}

void GrammarStreamingSession::reset() {
    fail();
    mel_.reset();
    enc_.reset();
    chunks_.reset();
    output_.reset();
    lifecycle_ = Lifecycle::active;
}

void run_stream_over_pcm(
    StreamingSession& sess, const ModelLoader& ml,
    const std::vector<float>& pcm16k,
    const std::function<void(const std::string&,
                             const std::vector<EouEvent>&,
                             const std::vector<Word>&)>& on_chunk,
    const std::string& target_lang) {
    // target_lang is intentionally unused here: the session already carries its
    // resolved prompt index from construction. The parameter exists so callers
    // can route a language through a single entry point (Phase 4 C-API/CLI).
    (void)target_lang;
    // 1. Full-clip mel [n_mels, T] (feat-major inner=T), matching the offline /
    //    NeMo online_normalization=False reference (normalization over the whole
    //    clip). The streaming numerics come from the carried encoder/decoder
    //    caches, not from chunking the front end.
    MelFrontend mel_fe(ml);
    std::vector<float> mel;
    int n_mels = 0, T = 0;
    mel_fe.compute(pcm16k, mel, n_mels, T);
    if (T <= 0) return;

    const int chunk0     = sess.chunk_size_first();      // 9
    const int chunk_main = sess.chunk_size();            // 16
    const int pre_cache  = sess.pre_encode_cache_size(); // 9

    // mel[:, lo:hi] in feat-major layout.
    auto window = [&](int lo, int hi) {
        const int len = hi - lo;
        std::vector<float> w((size_t)n_mels * len);
        for (int m = 0; m < n_mels; ++m)
            for (int t = 0; t < len; ++t)
                w[(size_t)m * len + t] = mel[(size_t)m * T + (lo + t)];
        return w;
    };

    int buffer_idx = 0;
    bool first = true;
    while (buffer_idx < T) {
        const int chunk_size = first ? chunk0 : chunk_main;
        const int shift      = chunk_size;  // shift_size == chunk_size here
        const int chunk_hi   = std::min(buffer_idx + chunk_size, T);
        if (chunk_hi - buffer_idx <= 0) break;
        const int lo = first ? buffer_idx : std::max(0, buffer_idx - pre_cache);
        std::vector<float> win = window(lo, chunk_hi);
        const int win_frames = chunk_hi - lo;
        const bool is_last = (chunk_hi >= T);

        sess.feed_mel_chunk(win, win_frames, is_last);

        if (on_chunk) {
            std::string nt = sess.take_new_text();
            std::vector<EouEvent> ev = sess.drain_events();
            std::vector<Word> wd = sess.drain_words();
            if (!nt.empty() || !ev.empty() || !wd.empty()) on_chunk(nt, ev, wd);
        }

        buffer_idx += shift;
        first = false;
    }
}

} // namespace pk
