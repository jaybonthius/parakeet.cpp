# C API (`libparakeet.so`)

The README has the overview and the ABI note. This page has the longer examples.
The current ABI version is 10 (`parakeet_capi_abi_version()`); all additions since v5 are additive.
The full symbol list and the exact signatures are in `include/parakeet_capi.h`.

`include/parakeet_capi.h` defines a flat, exception-free C-API meant for `dlopen` / FFI / LocalAI integration. Build the shared library with `-DPARAKEET_SHARED=ON`:

```c
#include "parakeet_capi.h"

parakeet_ctx *ctx = parakeet_capi_load("model.gguf");  // load ONCE
if (!ctx) { fprintf(stderr, "%s\n", parakeet_capi_last_error(ctx)); return 1; }

char *text = parakeet_capi_transcribe_path(ctx, "audio.wav", 0 /*default*/);
if (text) { printf("%s\n", text); parakeet_capi_free_string(text); }

parakeet_capi_free(ctx);
```

In-memory PCM:
```c
char *text = parakeet_capi_transcribe_pcm(ctx, samples, n_samples,
                                          sample_rate, 0 /*default*/);
```

Timestamps and confidence as JSON (matches NeMo `timestamps=True` + `max_prob`):
```c
char *json = parakeet_capi_transcribe_path_json(ctx, "audio.wav", 0 /*default*/);
// {"text":"...",
//  "frame_sec":0.080000,
//  "words":[{"w":"Well,","start":0.480,"end":0.640,"conf":0.7859}, ...],
//  "tokens":[{"id":639,"t":0.480,"conf":0.9969}, ...]}
if (json) { printf("%s\n", json); parakeet_capi_free_string(json); }
```
`start`/`end`/`t` are in seconds; `conf` is the rescaled softmax probability of the emitted token in `(0,1]` (a word's `conf` is the `min` over its tokens). `frame_sec` is the encoder frame stride in seconds (`hop x subsampling / sample_rate`); multiply a frame-unit segment gap threshold (NeMo's `segment_gap_threshold`) by it to get the seconds gap between words when forming segments.

## Streaming (cache-aware EOU model)

For `parakeet_realtime_eou_120m-v1`, a streaming session decodes 16 kHz mono f32 PCM as it arrives, returning newly-finalized text and signalling EOU/EOB events:

```c
parakeet_stream *s = parakeet_capi_stream_begin(ctx);
int eou = 0;
char *t = parakeet_capi_stream_feed(s, pcm, n_samples, &eou); // "" if none yet
if (t) { printf("%s", t); parakeet_capi_free_string(t); }
if (eou) printf(" [EOU]");
// ...feed more chunks...
char *tail = parakeet_capi_stream_finalize(s);                // flush the tail
if (tail) { printf("%s\n", tail); parakeet_capi_free_string(tail); }
parakeet_capi_stream_free(s);
```

`<EOU>` (end-of-utterance) and `<EOB>` (backchannel) are stripped from the text and surfaced via `*eou_out` (the CLI `--stream` prints them as `[EOU @ <t>s]` markers). The streaming transcript matches NeMo's cache-aware streaming exactly, and `finalize` flushes the end-of-stream tail without fabricating an `<EOU>` that NeMo would not emit.

## Buffered streaming (unified checkpoints, ABI v11)

A unified checkpoint (`nvidia/nemotron-labs-unified-asr-0.6b`) streams by re-encoding a window of audio at each step: `[left | chunk | right]` in 80 ms frames. The text it commits is final and is never revised. A buffered stream raises no `<EOU>`/`<EOB>`, so `eou` and `eob` stay 0 and `parakeet_capi_stream_drain_events` returns no events.

The `parakeet_capi_stream_*` calls above work for these models too. `parakeet_capi_stream_begin` and `_begin_lang` use the default window (`parakeet_capi_stream_presets_json` lists them, with `"default":true`). Use `parakeet_capi_stream_begin_opts` to choose another preset, or a custom window, and the language:

```c
parakeet_stream_opts o;
parakeet_capi_stream_opts_default(&o);
o.target_lang = "de";                       // NULL or "" = the model default
o.preset = PARAKEET_STREAM_PRESET_CUSTOM;   // or an index from the presets JSON, or PARAKEET_STREAM_PRESET_DEFAULT
o.left_frames = 70; o.chunk_frames = 7; o.right_frames = 7;
parakeet_stream *s = parakeet_capi_stream_begin_opts(ctx, &o);
if (!s) { fprintf(stderr, "%s\n", parakeet_capi_last_error(ctx)); }
```

`parakeet_capi_stream_kind(ctx)` returns `PARAKEET_STREAM_KIND_BUFFERED` for these models and `PARAKEET_STREAM_KIND_CACHE_AWARE` for the EOU and nemotron-3.5 models. `parakeet_capi_stream_info_json(s)` reports the window and the latency. `parakeet_capi_stream_reset(s)` starts a new utterance on the same stream. `parakeet_stream_opts.size` must be `sizeof(parakeet_stream_opts)`, and a cache-aware model accepts only the default preset.

The text does not depend on the feed size: the same audio fed in blocks of 1, 160, 1280 or 7777 samples gives the same tokens (see `test_capi_stream_unified` and `test_unified_stream`, which need the checkpoint). The window per step is set by the preset: latency is `(chunk + right) x 0.08 s`. The default is 1.12 s (`70,7,7`). `left + chunk + right` must be at most 4000 frames (320 s), `chunk` at least 1 and the other two at least 0; a window outside these limits is refused with `NULL` and `parakeet_capi_last_error`.

**CPU speed.** On a laptop-class CPU only the default 1.12 s preset keeps up with real time. The shorter presets fall behind, because each step re-encodes the whole `left + chunk + right` window, so the cost per step does not shrink with the chunk. Measure the GPU before relying on a shorter preset there.

The LocalAI backend (in the LocalAI repo) dlopens `libparakeet.so` and uses these symbols directly: the offline `parakeet_capi_transcribe_*` / `parakeet_capi_transcribe_path_json` and the streaming `parakeet_capi_stream_*`. See `include/parakeet_capi.h` for the full API. The C++ streaming session (`pk::StreamingSession`) also exposes per-word timestamps and confidence as words finalize, via `drain_words()` alongside the EOU events, which the CLI `--stream --timestamps` path prints.

## Readable grammar constraints

The additive `parakeet_capi_*_grammar` and `parakeet_capi_grammar_stream_*`
functions restrict decoding, not post-filtering. Legacy functions remain
unconstrained. Offline PCM/path calls select `decoder=0` (architecture default,
**transducer for hybrids**), `1` (CTC), or `2` (TDT/RNNT). A missing head is an
error, never a CTC fallback. `beam_size` is 1..64, ignored by CTC. Language
prompts use the same `target_lang` convention as `*_lang`.

```c
parakeet_grammar *g = parakeet_capi_grammar_create(ctx,
    "root ::= action \" \" room\n"
    "action ::= \"open\" | \"close\"\n"
    "room ::= \"kitchen\" | \"office\"");
if (!g) { /* inspect parakeet_capi_last_error(ctx) */ return 1; }
char *text = NULL;
int status = parakeet_capi_transcribe_path_grammar(
    ctx, g, "command.wav", 0, 4, NULL, &text);
if (status == PARAKEET_GRAMMAR_OK) printf("%s\n", text);
parakeet_capi_free_string(text); /* NO_MATCH/ERROR leave text NULL */
parakeet_capi_grammar_free(g);
```

This is a **regular subset**, not recursive GBNF/CFG: UTF-8 quoted literals,
character ranges/negated classes, concatenation, alternatives, groups, `? * +`,
acyclic named rules, and `#` comments. Unsupported syntax (including recursion,
`{m,n}`, wildcard `.`), invalid UTF-8/byte-fallback vocabulary pieces and resource
limit exhaustion fail explicitly. Case, punctuation and whitespace match
exactly after U+2581 becomes space and exactly one initial space is removed per
unit. Compatible subword segmentations remain legal; control tokens are not
text. The header lists escapes and ownership details.

CTC chooses a best original frame alignment, not transcript posterior. TDT uses
a bounded length-normalized beam; RNNT uses a bounded Viterbi alignment beam,
not summed-prefix probabilities. Illegal text is excluded before pruning without
renormalizing model probabilities. Finite beams/symbol limits can discard valid
completions. `NO_MATCH` means no retained accepting completion, **not acoustic
rejection**; a narrow grammar can force unrelated speech into an accepted phrase.
No calibrated confidence or general accuracy claim follows from acceptance.

### Real grammar streaming

Use a native cache-aware RNNT model with frame-local mel (e.g. realtime EOU or
the prompt-conditioned Nemotron streaming model), not an offline TDT/hybrid.
`begin` refuses unsupported models/configurations. Feed 16 kHz mono f32 PCM in
arbitrary fragment sizes; frontend, encoder schedule and caches are incremental.

```c
parakeet_grammar_stream *s = parakeet_capi_grammar_stream_begin(ctx, g, 4, "en");
if (!s) { /* inspect ctx last_error; g still belongs to caller */ return 1; }
parakeet_capi_grammar_free(g); /* s retains the compiled grammar */
char *json = NULL;
int status = parakeet_capi_grammar_stream_feed_json(s, pcm, n_samples, &json);
/* Consume json only on OK/NO_MATCH; ERROR leaves it NULL. */
parakeet_capi_free_string(json);
/* ...feed more audio while OK... */
status = parakeet_capi_grammar_stream_finalize_json(s, &json);
parakeet_capi_free_string(json);
parakeet_capi_grammar_stream_free(s);
```

Each successful document has `partial` (replace the displayed provisional
suffix), `units` (append only these newly committed accepted text units) and
`events` (new real `eou`/`eob` markers). Do **not** append `partial` to a transcript.
Units contain `text`, `endpoint_id`, absolute 64-bit encoder `frame` and seconds
`t`; events contain `type`, token `id`, `frame`, `t`. EOB is a backchannel, not
end-of-turn. Endpoints require acceptance and restart the grammar immediately;
all later native frames still run, preserving the next provisional suffix.
Predictor reset happens at native-chunk end, matching legacy timing. Selected
chunk-final units are irreversible bounded-beam commitments; incompatible
histories are pruned, not globally optimized against future audio.

Finalize flushes the actual tail without inventing EOU. An accepted tail has
`endpoint_id=-1`, end-of-input position and no event. An incomplete remainder
returns `NO_MATCH` with JSON, preserving prior units and the incomplete partial.
Finalize is terminal for both OK and NO_MATCH. Any ERROR is terminal too: reset
before further feed/finalize, never replay PCM on an errored stream. Reset clears
all buffers, caches, search, output queues and time to a fresh origin. Output
queues are acknowledged only after successful JSON allocation.

### Grammar scores

A grammar turns any audio into some accepted phrase, so the caller needs a way
to tell a real match from a forced one. Grammar results carry four extra fields:

- `score`: log probability of the accepted path.
- `free_score`: log probability of the best path the same decoder finds over
  the same encoder output without the grammar. CTC takes each frame's best
  label (the exact best alignment), TDT runs its ordinary beam search, and RNNT
  runs the same bounded beam with every token allowed. The encoder runs once;
  only the decoder search is repeated.
- `frames`: number of encoder frames both scores cover.
- `alternatives`: up to `beam_size - 1` other distinct accepted texts, best
  first, each with the `score` of its path over the same frames. They come from
  the paths the search already kept (for CTC, the best path into each other
  accepting end state); nothing extra is searched. A plausible phrase can be
  missing because the beam dropped it.

All scores are raw sums of natural-log probabilities (not divided by length),
never above zero, and **not calibrated confidence**. Compare `score` with
`free_score` of the same result: equal or close means the grammar barely changed
what the model would have said; far below means the audio says something else.
`(score - free_score) / frames` puts results of different lengths on one scale.
Both searches are bounded, so `score` can be slightly above `free_score`. TDT
picks its winner by length-normalized score, like the plain call, so a TDT
alternative can have a higher raw score than the winner.

Offline, `parakeet_capi_transcribe_pcm_grammar_json` and
`parakeet_capi_transcribe_path_grammar_json` take the same arguments and run the
same search as the plain calls, and return the scores for the whole request:

```c
char *json = NULL;
int status = parakeet_capi_transcribe_path_grammar_json(
    ctx, g, "command.wav", 0, 8, NULL, &json);
/* OK: {"text":"open kitchen","score":-3.1,"free_score":-2.9,"frames":31,
        "alternatives":[{"text":"close kitchen","score":-14.0}]} */
parakeet_capi_free_string(json); /* NO_MATCH/ERROR leave json NULL */
```

In a grammar stream every unit carries the same fields. A unit's span runs
from the previous published unit (or begin/reset) to the end of the native
chunk that published it, so audio after the endpoint up to the chunk end is
included. Units published by the same chunk share one span and the same
`score`, `free_score` and `frames`. Measured on `tests/fixtures/speech.wav`
with the realtime EOU model: the true sentence gives `score == free_score`;
a phrase covering only the first words, or a forced command on thunder, gives
about -0.5 to -0.75 per frame. For unrelated audio the RNNT beam often keeps no
accepting path at all, which shows up as no unit or `NO_MATCH`.

Contexts outlive all handles/streams/calls. Handles are context-bound and
shareable across concurrent requests/streams; one caller at a time per stream.
Freeing a handle after begin is safe. Diagnostics are best-effort even on
allocation/locking failure; no grammar boundary lets exceptions cross C.
Grammar wiring into N-best, legacy timestamps/confidence, VAD/batch/SAS/scene
and CLI is intentionally absent. Real EOB audio and GPU grammar behavior have
not been validated; EOB endpoint semantics have synthetic coverage.
