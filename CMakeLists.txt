cmake_minimum_required(VERSION 3.10)  # Specify the minimum CMake version
project(Compression)  # Set the project name

# Find ASIO lib
find_package(asio REQUIRED)

# Find the Crow library
set(Crow_DIR /home/runner/work/Compression/Compression/Crow/build)
find_package(Crow REQUIRED)

# Find the matplotlib-cpp library
find_package(matplotlib-cpp REQUIRED)

# Add your executable
add_executable(compression main_code.cpp)  # Replace 'compression' with your desired executable name

# Link libraries
target_link_libraries(compression Crow::Crow matplotlib-cpp::matplotlib-cpp)
