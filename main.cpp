#include <iostream>
#include <iomanip>
#include <optional>
#include "./encoder/include/WordPiece.h"
#include "./inference/include/ORTWrapper.h"

/**
 * todo:
 * segment id is always 0 right now. figure out a solution.
 * tokens are truncated if there are too many of them. figure out a solution.
 */

int main() {
    try {
        nlp::encoder::WordPiece encoder(
            std::string(PROJECT_ROOT_PATH) + "/hf_model/tokenizer.json",
            "/model/vocab",
            true,
            true,
            true,
            true,
            128
        );

        const auto& vocab = encoder.get_vocab_list();
        std::unordered_map<std::string, int64_t> string_map = vocab.get_string_to_id_map();

        // std::cout << std::left << std::setw(20) << "Token" << " | " << "ID" << std::endl;
        // std::cout << std::string(30, '-') << std::endl;
        // for (const auto& [token, id] : string_map) {
        //     std::cout << std::left << std::setw(20) << token << " | " << id << std::endl;
        // }

        auto tokens = encoder.encode("Thé quick Browñ fox   jumps over \n the lázy dog!... here is another sentence with segment id 1!");

        std::cout << "\n--- Tokenization Results (" << tokens.size() << " tokens) ---\n";
        std::cout << std::left << std::setw(6) << "Index" << std::setw(10) << "ID" << std::setw(18) << "Token" << "\n";
        std::cout << std::string(35, '-') << "\n";
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::cout << std::left << std::setw(6) << i << std::setw(10) << tokens[i].id  << std::setw(18) << tokens[i].text << "\n";
        }

        // nlp::inference::ORTWrapper model(std::string(PROJECT_ROOT_PATH) + "/hf_model/model.onnx");
        // std::vector<float> results = model.run(tokens);
        //
        // for (float val : results) {
        //     std::cout << val << " ";
        // }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
