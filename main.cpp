#include <iostream>
#include <iomanip>

#include "./tokenizer/include/WordPiece.h"
#include "./inference/include/OnnxEngine.h"

/**
 * todo:
 * segment id is always 0 right now. figure out a solution. https://medium.com/artiwise-nlp/text-segmentation-and-its-applications-to-aspect-based-sentiment-analysis-fb115f9ab4e9
 * tokens are truncated if there are too many of them. figure out a solution.
 * handle .bin/.h5 files?
 */

using ConfigMap = std::unordered_map<std::string, std::variant<std::string, int, bool>>;

int main() {
    try {
        ConfigMap hf_model = {
            {"config_path", std::string(PROJECT_ROOT_PATH) + "/onnx_models/sentence-transformers-all-mini-lm-l6-v2/tokenizer.json"},
            {"model_path", std::string(PROJECT_ROOT_PATH) + "/onnx_models/sentence-transformers-all-mini-lm-l6-v2/model.onnx"},
            {"vocab_key", std::string("/model/vocab")},
            {"clean_text", true},
            {"to_lowercase", true},
            {"strip_accents", true},
            {"handle_chinese_chars", true},
            {"max_input_chars_per_word", 100},
            {"max_length", 128}
        };

        const nlp::tokenizer::WordPiece tokenizer(
            std::get<std::string>(hf_model["config_path"]),
            std::get<std::string>(hf_model["vocab_key"]),
            std::get<bool>(hf_model["clean_text"]),
            std::get<bool>(hf_model["to_lowercase"]),
            std::get<bool>(hf_model["strip_accents"]),
            std::get<bool>(hf_model["handle_chinese_chars"]),
            std::get<int>(hf_model["max_input_chars_per_word"]),
            std::get<int>(hf_model["max_length"])
        );

        // const auto& vocab = encoder.get_vocab_list();
        // std::unordered_map<std::string, int64_t> string_map = vocab.get_string_to_id_map();
        // std::cout << std::left << std::setw(20) << "Token" << " | " << "ID" << std::endl;
        // std::cout << std::string(30, '-') << std::endl;
        // for (const auto& [token, id] : string_map) {
        //     std::cout << std::left << std::setw(20) << token << " | " << id << std::endl;
        // }

        const std::string text = "The weather is great!";
        const auto tokens = tokenizer.tokenize(text);

        std::cout << "\n--- Tokenization Results (" << tokens.size() << " tokens) ---\n";
        std::cout << std::left << std::setw(6) << "Index" << std::setw(10) << "ID" << std::setw(18) << "Token" << "\n";
        std::cout << std::string(35, '-') << "\n";
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::cout << std::left << std::setw(6) << i << std::setw(10) << tokens[i].id  << std::setw(18) << tokens[i].text << "\n";
        }

        cnlp::inference::OnnxEngine engine(
            std::get<std::string>(hf_model["model_path"])
        );

        std::vector<std::vector<float>> embeddings = engine.encode(tokens);

        std::cout << "--- Token Embeddings Preview ---" << std::endl;

        for (size_t i = 0; i < tokens.size(); ++i) {
            // Print the token text so we know which vector we're looking at
            std::cout << "Token [" << i << "] (" << tokens[i].text << "): [";

            const auto& vec = embeddings[i];
            size_t preview_size = 5;

            // Print first few values
            for (size_t j = 0; j < std::min(vec.size(), preview_size); ++j) {
                printf("% .4f", vec[j]); // Neat formatting with 4 decimal places
                if (j < preview_size - 1) std::cout << ", ";
            }

            if (vec.size() > preview_size) {
                std::cout << " ... " << vec.back();
            }

            std::cout << "] (Dim: " << vec.size() << ")" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
