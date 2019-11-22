cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)
set(CUSTOMFLAGS " -g -v")
set(REQUIRED_FLAGS "-Wall -Wextra -Werror")


set(SANATIZER "-fsanitize=address -fsanitize=undefined -g")
set(SANATIZER_THREAD "-fsanitize=thread -fsanitize=undefined -g")

set(CMAKE_CXX_FLAGS_DEBUG "${SANATIZER} ${CMAKE_CXX_FLAGS_DEBUG} ${REQUIRED_FLAGS}")
set(CMAKE_LINKER_FLAGS_DEBUG "${SANATIZER} ${CMAKE_LINKER_FLAGS_DEBUG} ${REQUIRED_FLAGS}")

project(PartA)
find_package(OpenCV REQUIRED)


add_executable(robotVision main.cpp)
target_link_libraries(robotVision ${OpenCV_LIBS})