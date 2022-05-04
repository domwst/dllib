cmake_minimum_required(VERSION 3.5)

get_filename_component(LIB_INCLUDE_PATH ".." ABSOLUTE)
get_filename_component(LIB_PATH "." ABSOLUTE)

file(GLOB_RECURSE LIB_HEADERS ${LIB_PATH}/*.hpp)

add_library(dllib INTERFACE ${LIB_HEADERS})
target_include_directories(dllib INTERFACE ${LIB_INCLUDE_PATH})
