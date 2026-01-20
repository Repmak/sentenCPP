#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <sentenCPP/tokenizer/TokenizerInterface.h>

namespace sentencpp::inference {

    class InferenceInterface {
        public:
            virtual ~InferenceInterface() = default;

            // Encodes Token objects into their vector embeddings.
            [[nodiscard]] virtual std::vector<std::vector<float>> encode(const std::vector<tokenizer::Token>& tokens) = 0;
    };

} // namespace sentencpp::inference
