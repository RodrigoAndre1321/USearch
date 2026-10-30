
cmake_minimum_required(VERSION 3.20)

project(usearch_abs_experiments LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(baseline_search baseline_search.cpp)

target_include_directories(baseline_search PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
)
add_executable(benchmark_hnsw benchmark_hnsw.cpp)

target_include_directories(benchmark_hnsw PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
)