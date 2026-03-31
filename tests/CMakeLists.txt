set(CMAKE_CXX_STANDARD 11)

# add_executable(test_nsg_index test_nsg_index.cpp)
# target_link_libraries(test_nsg_index ${PROJECT_NAME} -ltcmalloc)

# add_executable(test_nsg_search test_nsg_search.cpp)
# target_link_libraries(test_nsg_search ${PROJECT_NAME})

# add_executable(test_nsg_optimized_search test_nsg_optimized_search.cpp)
# target_link_libraries(test_nsg_optimized_search ${PROJECT_NAME} -ltcmalloc)

add_executable(test_alphacng_index test_alphacng_index.cpp)
target_link_libraries(test_alphacng_index ${PROJECT_NAME} -ltcmalloc)

add_executable(test_alphacng_search test_alphacng_search.cpp)
target_link_libraries(test_alphacng_search ${PROJECT_NAME} -ltcmalloc)
