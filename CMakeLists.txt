cmake_minimum_required(VERSION 3.20)
project(word-piece-tokenizer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(word-piece-tokenizer
        main.cpp
        tokenizer/src/ByteLevelBPE.cpp
        tokenizer/src/WordPiece.cpp
        tokenizer/src/VocabList.cpp
        tokenizer/include/ByteLevelBPE.h
        tokenizer/include/ITokenizer.h
        tokenizer/include/VocabList.h
        tokenizer/include/WordPiece.h
)

target_include_directories(word-piece-tokenizer PRIVATE tokenizer/include)
