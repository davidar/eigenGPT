cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# create project
project(MyProject)
set(CMAKE_CXX_STANDARD 17)

# add executable
add_executable(main main.cpp)

# add dependencies
include(cmake/CPM.cmake)

CPMAddPackage("gh:fmtlib/fmt#10.0.0")
CPMAddPackage("gh:gabime/spdlog@1.11.0")
CPMAddPackage("gh:nlohmann/json@3.11.2")
CPMAddPackage("gl:libeigen/eigen#3.4.0")

# link dependencies
target_link_libraries(main fmt::fmt spdlog::spdlog nlohmann_json::nlohmann_json Eigen3::Eigen)
