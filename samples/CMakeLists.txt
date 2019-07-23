include_directories(${CMAKE_SOURCE_DIR})

if(BUILD_SAMPLES)
    find_package(PythonLibs 2.7)
    
    if(NOT DEFINED PYTHONLIBS_FOUND)
        message("PythonLibs required to plot: sudo apt-get install python-matplotlib python-numpy python2.7-dev")
    else()
        include_directories(${PYTHON_INCLUDE_DIRS})
        add_definitions(-DPLOT)
    endif()

    add_executable(linear_aircraft linear_aircraft.cpp)
    target_link_libraries(linear_aircraft bflib ${PYTHON_LIBRARIES})

    add_executable(non_linear_aircraft non_linear_aircraft.cpp)
    target_link_libraries(non_linear_aircraft bflib ${PYTHON_LIBRARIES})

    add_executable(sine sine.cpp)
    target_link_libraries(sine bflib ${PYTHON_LIBRARIES})

    add_executable(pendulum pendulum.cpp)
    target_link_libraries(pendulum bflib ${PYTHON_LIBRARIES})

    add_executable(robot_localization robot_localization.cpp)
    target_link_libraries(robot_localization bflib ${PYTHON_LIBRARIES})
endif(BUILD_SAMPLES)