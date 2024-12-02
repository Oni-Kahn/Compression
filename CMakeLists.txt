cmake_minimum_required(VERSION 3.10)  # Specify the minimum CMake version
project(Compression)  # Set the project name

# Find ASIO lib
ind_path(asio_INCLUDE_DIR asio.hpp
  PATHS /usr/include
        /usr/local/include
)

find_library(asio_LIBRARY NAMES asio
  PATHS /usr/lib
        /usr/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(asio DEFAULT_MSG asio_LIBRARY asio_INCLUDE_DIR)

mark_as_advanced(asio_INCLUDE_DIR asio_LIBRARY)

# Find the Crow library
set(Crow_DIR /home/runner/work/Compression/Compression/Crow/build)
find_package(Crow REQUIRED)

# Find the matplotlib-cpp library
find_package(matplotlib-cpp REQUIRED)

# Add your executable
add_executable(Compression main_code.cpp)  # Replace 'compression' with your desired executable name

# Link libraries
target_link_libraries(Compression Crow::Crow matplotlib-cpp::matplotlib-cpp asio::asio)
