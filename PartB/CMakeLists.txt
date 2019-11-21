cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)
set(CUSTOMFLAGS " -g -v")
set(SANATIZER "-fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined -g")
set(SANATIZER_THREAD "-fsanitize=thread -fsanitize=undefined -g")
set(REQUIRED_FLAGS "-Wall -Wextra -Werror -pthread")


set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${REQUIRED_FLAGS} ${SANATIZER} ${CUSTOM_FLAGS}")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG}${REQUIRED_FLAGS} ${SANATIZER} ${CUSTOM_FLAGS}")

project(PartB)
find_package(OpenCV REQUIRED)


add_executable(displayImage hw3_b.cpp)
target_link_libraries(DisplayImage ${OpenCV_LIBS})