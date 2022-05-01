cmake_minimum_required(VERSION 3.5)

include(FetchContent)

# --------------------------------------------------------------------

# set(FETCHCONTENT_FULLY_DISCONNECTED ON)
# set(FETCHCONTENT_QUIET OFF)

# --------------------------------------------------------------------

FetchContent_Declare(
        ut
        GIT_REPOSITORY https://github.com/boost-ext/ut.git
        GIT_TAG v1.1.9
)
FetchContent_MakeAvailable(ut)

