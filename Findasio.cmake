find_path(asio_INCLUDE_DIR asio.hpp
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
