cmake_minimum_required(VERSION 2.8)
project(bflib)

option(BUILD_SAMPLES "Build the library samples" ON)

include(GNUInstallDirs)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(
${PROJECT_NAME}
INTERFACE $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/bflib>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries (${PROJECT_NAME} INTERFACE Eigen3::Eigen)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_11)

install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}_Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION
    ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

install(EXPORT ${PROJECT_NAME}_Targets
    FILE ${PROJECT_NAME}Targets.cmake
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/bflib DESTINATION include)

get_directory_property(has_parent PARENT_DIRECTORY)
if(NOT has_parent)
    # Samples
    if(BUILD_SAMPLES)
        add_subdirectory(samples)  
    endif(BUILD_SAMPLES) 
endif()