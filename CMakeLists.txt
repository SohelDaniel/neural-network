# Build for the from-scratch MNIST network and its tests.
cmake_minimum_required(VERSION 3.16)
project(neural_network LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# training is unbearably slow at -O0, so default to an optimized build
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
	add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# core library shared by the trainer and the tests
add_library(nn_core
	src/matrix.cpp
	src/layer.cpp
	src/network.cpp
	src/mnist.cpp
)
target_include_directories(nn_core PUBLIC src)

add_executable(nn src/main.cpp)
target_link_libraries(nn PRIVATE nn_core)
