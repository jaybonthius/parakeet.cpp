#include "parakeet_capi.h"
#include "audio_io.hpp"
#include "ggml_graph.hpp"
#include "stream_clips.hpp"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <locale>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
class DecimalPoint : public std::numpunct<char> {
public:
    explicit DecimalPoint(char value) : value_(value) {}
private:
    char do_decimal_point() const override { return value_; }
    char value_;
};
class NumericLocale {
public:
    NumericLocale() : c_locale_(std::setlocale(LC_NUMERIC, nullptr)), cpp_locale_() {}
    ~NumericLocale() {
        std::locale::global(cpp_locale_);
        std::setlocale(LC_NUMERIC, c_locale_.c_str());
    }
private:
    std::string c_locale_;
    std::locale cpp_locale_;
};
void check_locale(char decimal) {
    require(std::localeconv()->decimal_point[0] == decimal &&
            std::use_facet<std::numpunct<char>>(std::locale()).decimal_point() == decimal,
            "grammar call changed numeric locale");
}
void emit(const char* label, int status, char* json, char decimal) {
    std::unique_ptr<char, decltype(&parakeet_capi_free_string)> owned(json, parakeet_capi_free_string);
    require(status == PARAKEET_GRAMMAR_OK && json, "grammar JSON call failed");
    check_locale(decimal);
    std::printf("%s\t%s\n", label, json);
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected numeric locale");
        const char* path = std::getenv("PARAKEET_TEST_GGUF_EOU");
        if (!path) return 77;
        pk::set_num_threads(2);
        using Context = std::unique_ptr<parakeet_ctx, decltype(&parakeet_capi_free)>;
        using Grammar = std::unique_ptr<parakeet_grammar, decltype(&parakeet_capi_grammar_free)>;
        using Stream = std::unique_ptr<parakeet_grammar_stream, decltype(&parakeet_capi_grammar_stream_free)>;
        Context ctx(parakeet_capi_load(path), parakeet_capi_free);
        require(static_cast<bool>(ctx), "load EOU model");
        Grammar grammar(parakeet_capi_grammar_create(ctx.get(),
            "root ::= \"well i don't wish to see it any more observed phoebe turning away her eyes it is certainly very like the old portrait\""),
            parakeet_capi_grammar_free);
        require(static_cast<bool>(grammar), "compile speech grammar");
        Stream stream(parakeet_capi_grammar_stream_begin(ctx.get(), grammar.get(), 4, nullptr),
                      parakeet_capi_grammar_stream_free);
        require(static_cast<bool>(stream), "begin stream");
        pk::Audio audio;
        require(pk::load_audio_16k_mono("tests/fixtures/speech.wav", audio), "load speech");
        const std::string previous_c = std::setlocale(LC_NUMERIC, nullptr);
        const std::locale previous_cpp;
        {
            NumericLocale restore;
            if (!std::setlocale(LC_NUMERIC, argv[1])) {
                std::fprintf(stderr, "numeric locale unavailable: %s\n", argv[1]);
                return 77;
            }
            const char decimal = std::string(argv[1]) == "C" ? '.' : ',';
            std::locale::global(std::locale(std::locale::classic(), new DecimalPoint(decimal)));
            check_locale(decimal);
            auto run = [&](const char* label, const std::vector<float>& pcm) {
                require(parakeet_capi_grammar_stream_reset(stream.get()) == PARAKEET_GRAMMAR_OK, "reset stream");
                for (size_t pos = 0; pos < pcm.size(); pos += 1600) {
                    char* json = nullptr;
                    const int status = parakeet_capi_grammar_stream_feed_json(stream.get(), pcm.data() + pos,
                        static_cast<int>(std::min<size_t>(1600, pcm.size() - pos)), &json);
                    emit(label, status, json, decimal);
                }
                char* json = nullptr;
                const int status = parakeet_capi_grammar_stream_finalize_json(stream.get(), &json);
                emit(label, status, json, decimal);
            };
            run("utterances", pktest::two_utterance_clip(audio.samples));
            run("tail", audio.samples);
        }
        require(previous_c == std::setlocale(LC_NUMERIC, nullptr) && previous_cpp == std::locale(),
                "test did not restore numeric locales");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_capi_grammar_locale: %s\n", e.what());
        return 1;
    }
}
