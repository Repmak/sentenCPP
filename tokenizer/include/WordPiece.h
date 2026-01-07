#pragma once

#include <string>
#include <vector>
#include <optional>
#include <iostream>

#include "ITokenizer.h"

namespace nlp::tokenizer {

    class WordPiece : public ITokenizer {
        public:
            WordPiece(
                const std::string& config_path,
                const std::string& vocab_key,
                bool lowercase,
                bool clean_text,
                bool handle_chinese_chars,
                std::size_t max_length
            );

            const VocabList& get_vocab_list() const { return *vocab_list_; }

            [[nodiscard]] std::vector<Token> encode(std::string_view text) const override;
            [[nodiscard]] TokenRole identify_special_token(uint32_t id) const override;
            [[nodiscard]] size_t get_vocab_size() const override { return vocab_list_->size(); }

        private:
            std::unique_ptr<VocabList> vocab_list_;
            bool lowercase;
            bool clean_text;
            bool handle_chinese_chars;
            std::size_t max_length;

            void build_byte_encoder();

            [[nodiscard]] std::vector<std::string> bpe_merge(std::string_view word) const;

            // Splits text by regex (eg: whitespace, punctuation).
            [[nodiscard]] std::vector<std::string_view> pre_tokenize(std::string_view text) const;
    };

} // namespace nlp::tokenizer
