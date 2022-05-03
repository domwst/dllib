cmake_minimum_required(VERSION 3.5)
project(dllib)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")

include(cmake/sanitizers.cmake)

add_compile_options(-Wall -Wextra -Wpedantic -Werror -g -fno-omit-frame-pointer)

add_subdirectory(third_party)
add_subdirectory(dllib)

if (DLLIB_DEVELOPER)
    add_subdirectory(tests)
endif()
