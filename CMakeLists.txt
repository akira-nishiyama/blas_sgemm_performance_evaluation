cmake_minimum_required(VERSION 3.12)
project(blas_performance_comparison CXX)

option(BUILD_CLOOP    "build c loop evaluation."   ON)
option(BUILD_OPENMP   "build openmp evaluation."   ON)
option(BUILD_OPENBLAS "build openblas evaluation." ON)
option(BUILD_CUBLA    "build cubla evaluation."    ON)
set(LOOP_NUM 10 CACHE STRING "number of evaluation loop.")

message(STATUS "BUILD_CLOOP   : ${BUILD_CLOOP}")
message(STATUS "BUILD_OPENMP  : ${BUILD_OPENMP}")
message(STATUS "BUILD_OPENBLAS: ${BUILD_OPENBLAS}")
message(STATUS "BUILD_CUBLA   : ${BUILD_CUBLA}")
message(STATUS "LOOP_NUM      : ${LOOP_NUM}")

add_custom_target(eval_all ALL)

#test data generation
add_executable(gen_testdata gen_testdata.cpp)
target_link_libraries(gen_testdata openblas)

add_custom_command(
  OUTPUT c_10000.txt c_1000.txt
  COMMAND ./gen_testdata 100 1000 100
  COMMAND ./gen_testdata 2000 10000 1000
  DEPENDS gen_testdata
  )

add_custom_target(exec_gen_testdata ALL
  DEPENDS c_10000.txt)

#c loop
if(BUILD_CLOOP)
add_executable(sgemm_c sgemm.cpp)
target_link_libraries(sgemm_c openblas)
target_compile_definitions(sgemm_c PRIVATE C_LOOP)
add_custom_target(eval_c
  COMMAND sgemm_c 100 ${LOOP_NUM}
  COMMAND sgemm_c 200 ${LOOP_NUM}
  COMMAND sgemm_c 300 ${LOOP_NUM}
  COMMAND sgemm_c 400 ${LOOP_NUM}
  COMMAND sgemm_c 500 ${LOOP_NUM}
  COMMAND sgemm_c 600 ${LOOP_NUM}
  COMMAND sgemm_c 700 ${LOOP_NUM}
  COMMAND sgemm_c 800 ${LOOP_NUM}
  COMMAND sgemm_c 900 ${LOOP_NUM}
  COMMAND sgemm_c 1000 ${LOOP_NUM}
  DEPENDS c_1000.txt sgemm_c)
add_dependencies(eval_all eval_c)
endif()

#openmp
if(BUILD_OPENMP)
find_package(OpenMP)
add_executable(sgemm_openmp sgemm.cpp)
target_link_libraries(sgemm_openmp ${OpenMP_CXX_LIBRARIES} openblas)
target_compile_definitions(sgemm_openmp PRIVATE OPENMP)
target_compile_options(sgemm_openmp PRIVATE ${OpenMP_CXX_FLAGS})
add_custom_target(eval_openmp
  COMMAND sgemm_openmp 100 ${LOOP_NUM}
  COMMAND sgemm_openmp 200 ${LOOP_NUM}
  COMMAND sgemm_openmp 300 ${LOOP_NUM}
  COMMAND sgemm_openmp 400 ${LOOP_NUM}
  COMMAND sgemm_openmp 500 ${LOOP_NUM}
  COMMAND sgemm_openmp 600 ${LOOP_NUM}
  COMMAND sgemm_openmp 700 ${LOOP_NUM}
  COMMAND sgemm_openmp 800 ${LOOP_NUM}
  COMMAND sgemm_openmp 900 ${LOOP_NUM}
  COMMAND sgemm_openmp 1000 ${LOOP_NUM}
  DEPENDS c_1000.txt sgemm_openmp)
add_dependencies(eval_all eval_openmp)
endif()

#openblas
if(BUILD_OPENBLAS)
add_executable(sgemm_openblas sgemm.cpp)
target_link_libraries(sgemm_openblas openblas)
target_compile_definitions(sgemm_openblas PRIVATE OPENBLAS)
add_custom_target(eval_openblas
  COMMAND sgemm_openblas 100 ${LOOP_NUM}
  COMMAND sgemm_openblas 200 ${LOOP_NUM}
  COMMAND sgemm_openblas 300 ${LOOP_NUM}
  COMMAND sgemm_openblas 400 ${LOOP_NUM}
  COMMAND sgemm_openblas 500 ${LOOP_NUM}
  COMMAND sgemm_openblas 600 ${LOOP_NUM}
  COMMAND sgemm_openblas 700 ${LOOP_NUM}
  COMMAND sgemm_openblas 800 ${LOOP_NUM}
  COMMAND sgemm_openblas 900 ${LOOP_NUM}
  COMMAND sgemm_openblas 1000 ${LOOP_NUM}
  COMMAND sgemm_openblas 2000 ${LOOP_NUM}
  COMMAND sgemm_openblas 3000 ${LOOP_NUM}
  COMMAND sgemm_openblas 4000 ${LOOP_NUM}
  COMMAND sgemm_openblas 5000 ${LOOP_NUM}
  COMMAND sgemm_openblas 6000 ${LOOP_NUM}
  COMMAND sgemm_openblas 7000 ${LOOP_NUM}
  COMMAND sgemm_openblas 8000 ${LOOP_NUM}
  COMMAND sgemm_openblas 9000 ${LOOP_NUM}
  COMMAND sgemm_openblas 10000 ${LOOP_NUM}
  DEPENDS c_10000.txt c_1000.txt sgemm_openblas)
add_dependencies(eval_all eval_openblas)
endif()

#cuda
if(BUILD_CUBLA)

find_package(CUDA REQUIRED)
cuda_add_executable(sgemm_cublas sgemm.cpp)
target_compile_definitions(sgemm_cublas PRIVATE CUBLAS)
target_link_libraries(sgemm_cublas ${CUDA_cublas_LIBRARY} openblas)
add_custom_target(eval_cublas
  COMMAND sgemm_cublas 100 ${LOOP_NUM}
  COMMAND sgemm_cublas 200 ${LOOP_NUM}
  COMMAND sgemm_cublas 300 ${LOOP_NUM}
  COMMAND sgemm_cublas 400 ${LOOP_NUM}
  COMMAND sgemm_cublas 500 ${LOOP_NUM}
  COMMAND sgemm_cublas 600 ${LOOP_NUM}
  COMMAND sgemm_cublas 700 ${LOOP_NUM}
  COMMAND sgemm_cublas 800 ${LOOP_NUM}
  COMMAND sgemm_cublas 900 ${LOOP_NUM}
  COMMAND sgemm_cublas 1000 ${LOOP_NUM}
  COMMAND sgemm_cublas 2000 ${LOOP_NUM}
  COMMAND sgemm_cublas 3000 ${LOOP_NUM}
  COMMAND sgemm_cublas 4000 ${LOOP_NUM}
  COMMAND sgemm_cublas 5000 ${LOOP_NUM}
  COMMAND sgemm_cublas 6000 ${LOOP_NUM}
  COMMAND sgemm_cublas 7000 ${LOOP_NUM}
  COMMAND sgemm_cublas 8000 ${LOOP_NUM}
  COMMAND sgemm_cublas 9000 ${LOOP_NUM}
  COMMAND sgemm_cublas 10000 ${LOOP_NUM}
  DEPENDS c_10000.txt c_1000.txt sgemm_cublas)
add_dependencies(eval_all eval_cublas)

endif(BUILD_CUBLA)

#message(STATUS "*** dump start cmake variables ***")
#get_cmake_property(_variableNames VARIABLES)
#foreach(_variableName ${_variableNames})
#  message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()
#message(STATUS "*** dump end ***")



