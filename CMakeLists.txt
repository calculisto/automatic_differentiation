cmake_minimum_required (VERSION 3.25)

project (calculisto_automatic_differentiation
    VERSION 0.1.0
    DESCRIPTION "Forward-mode automatic differentiation with dual numbers in C++20"
    LANGUAGES CXX
)

include (GNUInstallDirs)

# The dependencies: the installed version if it is compatible, otherwise
# downloaded.
include (FetchContent)
FetchContent_Declare (calculisto_array
    GIT_REPOSITORY https://github.com/calculisto/array.git
    GIT_TAG        0.1.1
    GIT_SHALLOW    TRUE
    FIND_PACKAGE_ARGS 0.1
)
FetchContent_MakeAvailable (calculisto_array)

# The library: header-only.
add_library (calculisto_automatic_differentiation INTERFACE)
add_library (calculisto::automatic_differentiation ALIAS calculisto_automatic_differentiation)
set_target_properties (calculisto_automatic_differentiation PROPERTIES
    EXPORT_NAME automatic_differentiation
)
target_include_directories (calculisto_automatic_differentiation INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries (calculisto_automatic_differentiation INTERFACE calculisto::array)
target_compile_features (calculisto_automatic_differentiation INTERFACE cxx_std_20)

# The tests, only when this is the main project.
if (PROJECT_IS_TOP_LEVEL)
    include (CTest)
    if (BUILD_TESTING)
        add_subdirectory (tests)
    endif ()
endif ()
