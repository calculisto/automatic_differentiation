cmake_minimum_required (VERSION 3.25)

project (calculisto_automatic_differentiation
    VERSION 0.1.0
    DESCRIPTION
        "Forward-mode automatic differentiation with dual numbers in C++20"
    LANGUAGES CXX
)

include (GNUInstallDirs)

# Installation, for use with
# find_package (calculisto_automatic_differentiation): by default only when
# this is the main project, not when it is used by another one
# (add_subdirectory, FetchContent).
option (CALCULISTO_AUTOMATIC_DIFFERENTIATION_INSTALL
    "Generate the installation rules"
    ${PROJECT_IS_TOP_LEVEL}
)

# The library: header-only.
add_library (calculisto_automatic_differentiation INTERFACE)
add_library (calculisto::automatic_differentiation
    ALIAS calculisto_automatic_differentiation
)
set_target_properties (calculisto_automatic_differentiation PROPERTIES
    EXPORT_NAME automatic_differentiation
)
target_include_directories (calculisto_automatic_differentiation INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features (calculisto_automatic_differentiation INTERFACE
    cxx_std_20
)

if (CALCULISTO_AUTOMATIC_DIFFERENTIATION_INSTALL)
    include (CMakePackageConfigHelpers)
    # Header-only, so independent of the architecture: under share/.
    set (config_directory "${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME}")
    install (TARGETS calculisto_automatic_differentiation
        EXPORT calculisto_automatic_differentiation_targets
    )
    install (DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
    install (EXPORT calculisto_automatic_differentiation_targets
        NAMESPACE calculisto::
        FILE calculisto_automatic_differentiationTargets.cmake
        DESTINATION "${config_directory}"
    )
    configure_package_config_file (
        cmake/calculisto_automatic_differentiationConfig.cmake.in
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        INSTALL_DESTINATION "${config_directory}"
    )
    # Version 0.x: a new minor version may break the API.
    write_basic_package_version_file (
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        COMPATIBILITY SameMinorVersion
        ARCH_INDEPENDENT
    )
    install (FILES
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION "${config_directory}"
    )
endif ()

# The tests, only when this is the main project.
if (PROJECT_IS_TOP_LEVEL)
    include (CTest)
    if (BUILD_TESTING)
        add_subdirectory (tests)
    endif ()
endif ()
