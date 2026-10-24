// Compile the boundary here so linker wrapping targets only C-API allocations
// and diagnostic locks, never the inference library's allocator/mutex internals.
#include "../src/parakeet_capi.cpp"
#include "ggml_graph.hpp"

#include <cerrno>
#include <pthread.h>

namespace {
thread_local bool fail_malloc = false, fail_lock = false, fail_new = false;
thread_local bool unknown_exception = false, fail_terminal_json = false;
thread_local unsigned injected = 0;
thread_local long serialization_fail_after = -1;
thread_local parakeet_grammar_stream* watched = nullptr;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
}

extern "C" void* __real_malloc(size_t);
extern "C" void* __wrap_malloc(size_t size) {
    if (fail_malloc) { ++injected; return nullptr; }
    return __real_malloc(size);
}
extern "C" int __real_pthread_mutex_lock(pthread_mutex_t*);
extern "C" int __wrap_pthread_mutex_lock(pthread_mutex_t* mutex) {
    if (fail_lock) { ++injected; return EAGAIN; }
    return __real_pthread_mutex_lock(mutex);
}
void* operator new(std::size_t bytes) {
    const bool serialization_failure = serialization_fail_after >= 0 && serialization_fail_after-- == 0;
    if (serialization_failure) serialization_fail_after = -1;
    if (serialization_failure || fail_new || (fail_terminal_json && watched && !watched->session.active())) {
        ++injected;
        if (unknown_exception) throw 7;
        throw std::bad_alloc();
    }
    if (void* ptr = __real_malloc(bytes ? bytes : 1)) return ptr;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    try { return ::operator new(size); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    try { return ::operator new[](size); } catch (...) { return nullptr; }
}
void operator delete(void* ptr, const std::nothrow_t&) noexcept { std::free(ptr); }
void operator delete[](void* ptr, const std::nothrow_t&) noexcept { std::free(ptr); }
void operator delete(void* ptr) noexcept { std::free(ptr); }
void operator delete[](void* ptr) noexcept { std::free(ptr); }
void operator delete(void* ptr, std::size_t) noexcept { std::free(ptr); }
void operator delete[](void* ptr, std::size_t) noexcept { std::free(ptr); }

int main() {
    try {
        const char* path = std::getenv("PARAKEET_TEST_GGUF_EOU");
        if (!path) return 77;
        pk::set_num_threads(2);
        std::unique_ptr<parakeet_ctx, decltype(&parakeet_capi_free)> ctx(parakeet_capi_load(path), parakeet_capi_free);
        require(static_cast<bool>(ctx), "load model");
        parakeet_ctx unsupported;
        require(!parakeet_capi_grammar_create(&unsupported, "root ::= \"a\""), "non-ASR accepted");
        for (bool unknown : {false, true}) {
            injected = 0;
            unknown_exception = unknown;
            fail_new = true;
            auto* grammar = parakeet_capi_grammar_create(ctx.get(), "root ::= \"a\"");
            char sentinel;
            char* out = &sentinel;
            int rc = parakeet_capi_transcribe_pcm_grammar(ctx.get(), nullptr, nullptr, 0, 16000, 0, 4, nullptr, &out);
            fail_new = false;
            require(!grammar && rc == PARAKEET_GRAMMAR_ERROR && !out && injected >= 2, "allocation/unknown diagnostic exception escaped");
        }
        unknown_exception = false;
        std::unique_ptr<parakeet_grammar, decltype(&parakeet_capi_grammar_free)> empty(
            parakeet_capi_grammar_create(ctx.get(), "root ::= \"\""), parakeet_capi_grammar_free);
        require(static_cast<bool>(empty), "compile empty");
        for (bool locking : {false, true}) {
            char sentinel;
            char* out = &sentinel;
            injected = 0;
            fail_malloc = !locking; fail_lock = locking;
            int rc = parakeet_capi_transcribe_pcm_grammar(ctx.get(), empty.get(), nullptr, 0, 16000, 0, 4, nullptr, &out);
            fail_malloc = fail_lock = false;
            require(rc == PARAKEET_GRAMMAR_ERROR && !out && injected, "offline output/diagnostic ownership");
        }
        fail_lock = true;
        auto* created = parakeet_capi_grammar_create(ctx.get(), "root ::= \"\"");
        auto* begun = parakeet_capi_grammar_stream_begin(ctx.get(), empty.get(), 4, nullptr);
        fail_lock = false;
        require(!created && !begun, "success-clear failure leaked create/begin result");
        const char* rules = "root ::= \"well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait\"";
        std::unique_ptr<parakeet_grammar, decltype(&parakeet_capi_grammar_free)> grammar(
            parakeet_capi_grammar_create(ctx.get(), rules), parakeet_capi_grammar_free);
        require(static_cast<bool>(grammar), "compile speech grammar");
        std::unique_ptr<parakeet_grammar_stream, decltype(&parakeet_capi_grammar_stream_free)> stream(
            parakeet_capi_grammar_stream_begin(ctx.get(), grammar.get(), 4, nullptr), parakeet_capi_grammar_stream_free);
        require(static_cast<bool>(stream), "begin stream");
        pk::Audio audio;
        require(pk::load_audio_16k_mono("tests/fixtures/speech.wav", audio), "load speech");
        auto pcm = audio.samples;
        pcm.insert(pcm.end(), 9600, 0.0f);
        pcm.insert(pcm.end(), audio.samples.begin(), audio.samples.end());
        stream->session.feed(pcm.data(), static_cast<int>(pcm.size()));
        const auto reference = grammar_stream_json(stream->session);
        const auto queued = stream->session.units().size();
        const auto frame = stream->session.encoder_frame();
        require(queued > 0, "serializer sweep needs queued real events");
        for (bool unknown : {false, true}) {
            bool completed = false;
            for (long offset = 0; offset < 256; ++offset) {
                injected = 0;
                unknown_exception = unknown;
                serialization_fail_after = offset;
                bool threw = false;
                std::string serialized;
                try { serialized = grammar_stream_json(stream->session); }
                catch (...) { threw = true; }
                serialization_fail_after = -1;
                unknown_exception = false;
                require(stream->session.active() && stream->session.units().size() == queued &&
                        stream->session.encoder_frame() == frame, "serializer consumed queued output");
                if (!injected) {
                    require(!threw && serialized == reference, "serialization changed without injected failure");
                    std::printf("serializer allocation sweep (%s): %ld throwing sites passed\n",
                                unknown ? "unknown" : "bad_alloc", offset);
                    completed = true;
                    break;
                }
                require(injected == 1 && threw, "serializer swallowed allocation/unknown failure");
            }
            require(completed, "serializer allocation sweep exceeded test bound");
        }
        for (int failure : {0, 1}) {
            require(parakeet_capi_grammar_stream_reset(stream.get()) == PARAKEET_GRAMMAR_OK, "reset before mutable fault");
            char* out = nullptr;
            injected = 0;
            fail_malloc = failure == 0;
            fail_lock = failure == 1;
            int rc = parakeet_capi_grammar_stream_feed_json(stream.get(), pcm.data(), static_cast<int>(pcm.size()), &out);
            fail_malloc = fail_lock = false;
            require(rc == PARAKEET_GRAMMAR_ERROR && !out && injected, "post-inference feed fault escaped");
            require(stream->session.encoder_frame() > 98 && !stream->session.active() && !stream->session.units().empty(),
                    "mutable-work fault acknowledged/lost queued accepted unit");
            const auto pending = stream->session.units().size();
            require(parakeet_capi_grammar_stream_feed_json(stream.get(), nullptr, 0, &out) == PARAKEET_GRAMMAR_ERROR && !out &&
                    stream->session.units().size() == pending, "error stream replayed/consumed outputs");
        }
        for (int failure : {0, 1, 2}) {
            require(parakeet_capi_grammar_stream_reset(stream.get()) == PARAKEET_GRAMMAR_OK, "reset before final fault");
            char* out = nullptr;
            require(parakeet_capi_grammar_stream_feed_json(stream.get(), audio.samples.data(), static_cast<int>(audio.samples.size()), &out)
                == PARAKEET_GRAMMAR_OK && out, "speech feed");
            parakeet_capi_free_string(out);
            out = nullptr; injected = 0;
            fail_malloc = failure == 0;
            watched = stream.get();
            fail_terminal_json = failure != 0;
            unknown_exception = failure == 2;
            int rc = parakeet_capi_grammar_stream_finalize_json(stream.get(), &out);
            fail_malloc = fail_terminal_json = unknown_exception = false;
            watched = nullptr;
            require(rc == PARAKEET_GRAMMAR_ERROR && !out && injected && !stream->session.active() &&
                    stream->session.units().size() == 1 && stream->session.units()[0].endpoint_id == -1,
                    "final-tail allocation/serialization fault lost output or escaped");
        }
        fail_lock = true;
        int reset_rc = parakeet_capi_grammar_stream_reset(stream.get());
        fail_lock = false;
        require(reset_rc == PARAKEET_GRAMMAR_ERROR && !stream->session.active(), "reset success-clear fault not terminal");
        require(parakeet_capi_grammar_stream_reset(stream.get()) == PARAKEET_GRAMMAR_OK &&
                stream->session.encoder_frame() == 0 && stream->session.units().empty(), "fresh recovery after faults");
        char* out = nullptr;
        require(parakeet_capi_grammar_stream_feed_json(stream.get(), audio.samples.data(), static_cast<int>(audio.samples.size()), &out)
            == PARAKEET_GRAMMAR_OK && out, "recovery feed");
        parakeet_capi_free_string(out);
        require(parakeet_capi_grammar_stream_finalize_json(stream.get(), &out) == PARAKEET_GRAMMAR_OK && out &&
                std::strstr(out, "old portrait"), "recovery finalize");
        parakeet_capi_free_string(out);
        std::puts("C API grammar faults PASS: C allocation, serialization std/unknown exceptions, diagnostic locks, queued ownership, terminal/reset recovery");
        return 0;
    } catch (const std::exception& e) {
        serialization_fail_after = -1;
        fail_malloc = fail_lock = fail_new = fail_terminal_json = false;
        std::fprintf(stderr, "test_capi_grammar_faults: %s\n", e.what());
        return 1;
    } catch (...) {
        serialization_fail_after = -1;
        fail_malloc = fail_lock = fail_new = fail_terminal_json = false;
        std::fputs("test_capi_grammar_faults: exception crossed boundary\n", stderr);
        return 1;
    }
}
