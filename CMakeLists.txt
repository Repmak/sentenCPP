cmake_minimum_required(VERSION 3.20)
project(word-piece-tokenizer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ICU COMPONENTS uc i18n REQUIRED)

add_executable(word-piece-tokenizer
        main.cpp
        encoder/src/VocabList.cpp
        encoder/src/WordPiece.cpp
        encoder/include/IEncoder.h
        encoder/include/VocabList.h
        encoder/include/WordPiece.h
)

add_definitions(-DPROJECT_ROOT_PATH="${CMAKE_SOURCE_DIR}")

target_include_directories(word-piece-tokenizer PRIVATE
        encoder/include
        ${ICU_INCLUDE_DIRS}
)

if(TARGET ICU::uc)
    target_link_libraries(word-piece-tokenizer PRIVATE ICU::uc ICU::i18n)
else()
    target_link_libraries(word-piece-tokenizer PRIVATE ${ICU_LIBRARIES})
endif()
