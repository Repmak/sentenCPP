#include <iostream>
#include <iomanip>
#include "./tokenizer/include/WordPiece.h"

int main() {
    try {
        nlp::tokenizer::WordPiece tokenizer(
            "./hf_model/tokenizer.json",
            "/model/vocab",
            true,
            true,
            true,
            128
        );

        const auto& vocab = tokenizer.get_vocab_list();
        const auto& string_map = vocab.get_string_to_id_map();
        const auto& special_ids = vocab.get_special_ids();

        std::cout << std::left << std::setw(20) << "Token" << " | " << "ID" << std::endl;
        std::cout << std::string(30, '-') << std::endl;

        size_t count = 0;
        for (const auto& [token, id] : string_map) {
            if (count++ >= 100) break;
            std::cout << std::left << std::setw(20) << token << " | " << id << std::endl;
        }

        std::cout << "\n--- Special Token IDs ---" << std::endl;

        auto print_special = [](const std::string& name, std::uint32_t id) {
            std::cout << std::left << std::setw(15) << name << " : ";
            if (id) std::cout << id;
            else    std::cout << "None";
            std::cout << std::endl;
        };

        print_special("Padding", special_ids.padding);
        print_special("Unknown", special_ids.unknown);
        print_special("Classification", special_ids.classification);
        print_special("Separator", special_ids.separator);
        print_special("Mask", special_ids.mask);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
