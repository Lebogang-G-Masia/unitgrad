cmake_minimum_required(VERSION 3.14)
project(unitgrad)

add_library(unitgrad INTERFACE)

target_include_directories(unitgrad INTERFACE engine/)
