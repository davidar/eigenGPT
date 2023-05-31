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
