cmake_minimum_required( VERSION 2.8 )

# Create Project
#project( solution_feigang )


# Set StartUp Project (Option)
# (This setting is able to enable by using CMake 3.6.0 RC1 or later.)
#set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "project" )

# Find Packages
if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()
find_package( PCL 1.8 REQUIRED )
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/third" "${CMAKE_CURRENT_SOURCE_DIR}/header")
if( PCL_FOUND )
  # [C/C++]>[General]>[Additional Include Directories]
  include_directories( ${PCL_INCLUDE_DIRS} )

  # [C/C++]>[Preprocessor]>[Preprocessor Definitions]
  add_definitions( ${PCL_DEFINITIONS} )

  # For Use Not PreCompiled Features
  #add_definitions( -DPCL_NO_PRECOMPILE )

  # [Linker]>[General]>[Additional Library Directories]
  link_directories( ${PCL_LIBRARY_DIRS} )
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    target_link_libraries(${you_executable_name} OpenMP::OpenMP_CXX)
else()
    message(FATAL_ERROR "openmp not found!")
endif()
  # [Linker]>[Input]>[Additional Dependencies]
  set (test_SRCS   main.cpp thickness_cal.cpp)
  set (test_INCS	 thickness_cal.h)
  add_executable( project_feigang_hou ${test_SRCS} ${test_INCS} )
  target_link_libraries( project_feigang_hou ${PCL_LIBRARIES} )
  SET(CMAKE_BUILD_TYPE "Release")
endif()
