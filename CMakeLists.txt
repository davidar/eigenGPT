cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# create project
project(MyProject)

# add executable
add_executable(main main.cpp)

# add dependencies
include(cmake/CPM.cmake)

CPMAddPackage("gh:fmtlib/fmt#7.1.3")
CPMAddPackage("gh:nlohmann/json@3.10.5")
CPMAddPackage("gl:libeigen/eigen#3.4.0")

# link dependencies
target_link_libraries(main fmt::fmt nlohmann_json::nlohmann_json Eigen3::Eigen)
