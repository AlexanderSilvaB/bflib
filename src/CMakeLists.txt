find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_library(KFs INTERFACE)
target_include_directories(KFs INTERFACE KFs/)
target_link_libraries (KFs INTERFACE Eigen3::Eigen)

add_subdirectory(samples)   