cmake_minimum_required(VERSION 3.30)
project(x86Benchmarking C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -O3")
add_executable(x86Benchmarking main.c)
