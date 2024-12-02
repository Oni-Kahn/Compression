cmake_minimum_required(VERSION 3.10)  # Specify the minimum CMake version
project(Compression)  # Set the project name

# Find the Crow library
set(Crow_DIR /home/runner/work/Compression/Compression/Crow/build)
find_package(Crow REQUIRED)

# Find the matplotlib-cpp library
find_package(matplotlib-cpp REQUIRED)

# Find ASIO lib
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
find_package(asio REQUIRED)

# Add your executable
add_executable(Compression main_code.cpp)  # Replace 'compression' with your desired executable name

# Link libraries
target_link_libraries(Compression Crow::Crow matplotlib-cpp::matplotlib-cpp asio::asio)
