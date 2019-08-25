cmake_minimum_required(VERSION 2.8)

get_directory_property(has_parent PARENT_DIRECTORY)
if(NOT has_parent)
    add_subdirectory(../ bflib)
endif()


project(bflib_samples)
include_directories(..)

find_package(PythonLibs 2.7)
if(NOT DEFINED PYTHONLIBS_FOUND)
    message("PythonLibs required to plot: sudo apt-get install python-matplotlib python-numpy python2.7-dev")
    set(PYTHON_LIBRARIES "")
else()
    include_directories(${PYTHON_INCLUDE_DIRS})
    add_definitions(-DPLOT)
endif()

find_package(OpenCV)
if(NOT DEFINED OpenCV_FOUND)
    message("OpenCV required to real time plot: sudo apt-get install libopencv-dev")
    set(OpenCV_LIBS "")
else()
    include_directories(${OpenCV_INCLUDE_DIRS})
    add_definitions(-DPLOT_REALTIME)
endif()

add_executable(linear_aircraft linear_aircraft.cpp)
target_link_libraries(linear_aircraft bflib ${PYTHON_LIBRARIES})

add_executable(non_linear_aircraft non_linear_aircraft.cpp)
target_link_libraries(non_linear_aircraft bflib ${PYTHON_LIBRARIES})

add_executable(sine sine.cpp)
target_link_libraries(sine bflib ${PYTHON_LIBRARIES})

add_executable(pendulum pendulum.cpp)
target_link_libraries(pendulum bflib ${PYTHON_LIBRARIES})

add_executable(robot_localization_ekf robot_localization_ekf.cpp)
target_link_libraries(robot_localization_ekf bflib ${PYTHON_LIBRARIES} ${OpenCV_LIBS})

add_executable(robot_localization_mc robot_localization_mc.cpp)
target_link_libraries(robot_localization_mc bflib ${PYTHON_LIBRARIES} ${OpenCV_LIBS})
