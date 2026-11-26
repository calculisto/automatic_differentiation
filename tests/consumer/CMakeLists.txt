# A project using the installed library through find_package: built and run
# by the consumer test.
cmake_minimum_required (VERSION 3.25)

project (calculisto_automatic_differentiation_consumer LANGUAGES CXX)

find_package (calculisto_automatic_differentiation 0.1 REQUIRED)

add_executable (consumer main.cpp)
target_link_libraries (consumer PRIVATE calculisto::automatic_differentiation)
