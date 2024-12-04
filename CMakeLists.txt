cmake_minimum_required(VERSION 3.10)  # Specify the minimum CMake version
project(Compression)  # Set the project name

# Find the matplotlib-cpp library
set(matplotlib-cpp_DIR /home/runner/work/Compression/Compression/matplotlib-cpp/build)
set(matplotlib-cpp_CONFIG ${matplotlib-cpp_DIR}/matplotlib_cppConfig.cmake)
find_package(matplotlib_cpp REQUIRED CONFIG)
include_directories(matplotlib_cpp)

# Add your executable
add_executable(Compression main_code.cpp)  # Replace 'compression' with your desired executable name
