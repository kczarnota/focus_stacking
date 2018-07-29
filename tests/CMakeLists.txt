add_subdirectory(/usr/src/gtest
                 ${PROJECT_BINARY_DIR}/gtest)
include(CTest) # Include testing cmake package.
set(TEST_BINARY ${PROJECT_NAME}_test)

find_package( OpenCV REQUIRED )

include_directories(../src)
link_directories(../lib)

add_executable(${TEST_BINARY} tests.cpp)
target_link_libraries(${TEST_BINARY} fs gtest gtest_main ${OpenCV_LIBS})
add_test(NAME ${TEST_BINARY} COMMAND ${EXECUTABLE_OUTPUT_PATH}/${TEST_BINARY})