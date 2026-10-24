#include "parakeet_capi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    char sentinel;
    char* text = &sentinel;
    if (parakeet_capi_transcribe_pcm_grammar(NULL, NULL, NULL, 0, 16000, 0, 4, NULL, &text)
            != PARAKEET_GRAMMAR_ERROR || text != NULL) return 1;
    const char* path = getenv("PARAKEET_TEST_GGUF_EOU");
    if (!path) return 77;
    parakeet_ctx* ctx = parakeet_capi_load(path);
    if (!ctx) return 1;
    const char* expected = "well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait";
    parakeet_grammar* speech = parakeet_capi_grammar_create(ctx,
        "root ::= \"well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait\"");
    if (!speech) { parakeet_capi_free(ctx); return 1; }
    text = NULL;
    int status = parakeet_capi_transcribe_path_grammar(ctx, speech, "tests/fixtures/speech.wav", 0, 4, NULL, &text);
    int failed = status != PARAKEET_GRAMMAR_OK || text == NULL;
    if (text) failed |= strcmp(text, expected) != 0;
    parakeet_capi_free_string(text);
    parakeet_capi_grammar_free(speech);
    parakeet_grammar* grammar = parakeet_capi_grammar_create(ctx, "root ::= \"\"");
    if (!grammar) { parakeet_capi_free(ctx); return 1; }
    status = parakeet_capi_transcribe_pcm_grammar(ctx, grammar, NULL, 0, 16000, 0, 4, NULL, &text);
    failed |= status != PARAKEET_GRAMMAR_OK || text == NULL || (text && *text);
    parakeet_capi_free_string(text);
    text = &sentinel;
    status = parakeet_capi_transcribe_path_grammar(ctx, grammar, NULL, 0, 4, NULL, &text);
    failed |= status != PARAKEET_GRAMMAR_ERROR || text != NULL;
    parakeet_grammar_stream* stream = parakeet_capi_grammar_stream_begin(ctx, grammar, 4, NULL);
    parakeet_capi_grammar_free(grammar);
    if (!stream) { parakeet_capi_free(ctx); return 1; }
    text = NULL;
    status = parakeet_capi_grammar_stream_feed_json(stream, NULL, 0, &text);
    failed |= status != PARAKEET_GRAMMAR_OK || text == NULL;
    if (text) failed |= strcmp(text, "{\"partial\":\"\",\"units\":[],\"events\":[]}") != 0;
    parakeet_capi_free_string(text);
    text = NULL;
    status = parakeet_capi_grammar_stream_finalize_json(stream, &text);
    failed |= status != PARAKEET_GRAMMAR_OK || text == NULL;
    parakeet_capi_free_string(text);
    text = &sentinel;
    status = parakeet_capi_grammar_stream_feed_json(stream, NULL, 0, &text);
    failed |= status != PARAKEET_GRAMMAR_ERROR || text != NULL;
    failed |= parakeet_capi_grammar_stream_reset(stream) != PARAKEET_GRAMMAR_OK;
    parakeet_capi_grammar_stream_free(stream);
    parakeet_capi_free(ctx);
    parakeet_capi_grammar_stream_free(NULL);
    parakeet_capi_grammar_free(NULL);
    puts(failed ? "C grammar consumer FAIL" : "C grammar consumer PASS");
    return failed ? 1 : 0;
}
