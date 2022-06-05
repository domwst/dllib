cmake_minimum_required(VERSION 3.5)

message(STATUS "DLLib examples")

get_filename_component(EXAMPLES_PATH "." ABSOLUTE)

file(GLOB EXAMPLE_SOURCES RELATIVE ${EXAMPLES_PATH} "./*.cpp")

foreach(example_source_file ${EXAMPLE_SOURCES})
    string(REPLACE ".cpp" "" example_name ${example_source_file})
    set(target_name "${example_name}_example")
    add_executable(${target_name} ${example_source_file})
    target_link_libraries(${target_name} PUBLIC dllib)
endforeach()
