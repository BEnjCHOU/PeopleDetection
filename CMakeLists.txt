# Required by CMake 3.6
CMAKE_MINIMUM_REQUIRED(VERSION 3.6)
project(pclPeopleDetection)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package( PCL REQUIRED )
find_package(Boost REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(PeopleDetection PeopleDetection.cpp)

TARGET_LINK_LIBRARIES(pclDemo ${PCL_LIBRARIES} -lboost_system)