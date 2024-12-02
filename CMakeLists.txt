cmake_minimum_required(VERSION 3.10)  # Specify the minimum CMake version
project(Compression)  # Set the project name

# Find the matplotlib-cpp library
set(matplotlib-cpp_DIR /usr/local/include/matplotlibcpp.h)
find_package(matplotlib-cpp REQUIRED)

# Add your executable
add_executable(Compression main_code.cpp)  # Replace 'compression' with your desired executable name

# Link libraries
target_link_libraries(Compression matplotlib-cpp::matplotlib-cpp)
