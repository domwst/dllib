message(STATUS "DLLib tests")

# List sources

file(GLOB TEST_SOURCES "./*.cpp")

# All tests target

add_executable(dllib_all_tests ${TEST_SOURCES})
target_link_libraries(dllib_all_tests dllib ut)

