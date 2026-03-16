cmake_minimum_required(VERSION 3.14)
project(unitgrad)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(FetchContent)
FetchContent_Declare(
    lingebra
    GIT_REPOSITORY  https://github.com/Lebogang-G-Masia/lingebra.git
    GIT_TAG         main
)
FetchContent_MakeAvailable(lingebra)
add_library(unitgrad INTERFACE)
target_include_directories(unitgrad INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(unitgrad INTERFACE lingebra)
add_library(a a.cpp)
target_link_libraries(a PRIVATE unitgrad)
