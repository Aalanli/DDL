cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(test LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_executable(main conv.cu)
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 14)
set_property(TARGET main PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 75-real 75-virtual)
