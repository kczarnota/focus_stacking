find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)

add_library(fs focus_stacking.cpp image_loader.cpp)
target_link_libraries(fs ${Boost_LIBRARIES})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} fs)
