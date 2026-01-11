#pragma once

#include <string>
#include <vector>
#include <string_view>

#include "VocabList.h"

namespace nlp::tokenizer {

    struct Token {
        uint32_t id;          // The numerical ID according to the model.
        std::string text;     // The original string representation.
        size_t start_offset;  // Start position in the original string.
        size_t end_offset;    // End position in the original string.
        TokenRole type;       // For special tokens.
    };

    class ITokenizer {
        public:
            virtual ~ITokenizer() = default;

            // Returns the total vocabulary size.
            virtual size_t get_vocab_size() const = 0;

            // Checks if the token linked to the parameter id is a special token.
            virtual TokenRole identify_special_token(uint32_t id) const = 0;

            // Encodes raw text into a Token object.
            virtual std::vector<Token> tokenize(std::string_view text) const = 0;
    };

} // namespace nlp::tokenizer
