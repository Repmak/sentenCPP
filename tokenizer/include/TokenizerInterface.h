#pragma once

#include <string>
#include <vector>
#include <string_view>

#include "VocabList.h"

namespace nlp::tokenizer {

    struct Token {
        int64_t id;              // The numerical ID according to the model's vocabulary.
        std::string text;        // The original string representation (not strictly needed by the Onnx model).
        int64_t attention_mask;  // 1 for real tokens, 0 for padding.
        int64_t segment_id;      // Defines what sentence the token belongs to.
    };

    class TokenizerInterface {
        public:
            virtual ~TokenizerInterface() = default;

            // Returns the total vocabulary size.
            [[nodiscard]] virtual size_t get_vocab_size() const = 0;

            // Tokenize raw text into Token objects.
            [[nodiscard]] virtual std::vector<Token> tokenize(std::string_view text) const = 0;
    };

} // namespace nlp::tokenizer
