cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# create project
project(MyProject)
set(CMAKE_CXX_STANDARD 17)

add_custom_command(
    OUTPUT model.o
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/gpt2/"
    COMMAND ld -r -b binary -o "${CMAKE_CURRENT_BINARY_DIR}/model.o" model.safetensors
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gpt2/model.safetensors"
)

# add executable
add_executable(main main.cpp model.o)

# add dependencies
include(cmake/CPM.cmake)

CPMAddPackage("gh:fmtlib/fmt#10.0.0")
# CPMAddPackage("gh:gabime/spdlog@1.11.0")
CPMAddPackage("gh:nlohmann/json@3.11.2")
# CPMAddPackage("gl:libeigen/eigen#3.4.0")

add_subdirectory(gpt2-tokenizer)

# link dependencies
target_link_libraries(main PRIVATE
    # Eigen3::Eigen
    fmt::fmt
    nlohmann_json::nlohmann_json
    re2::re2
    # spdlog::spdlog
    tokenizer
)
