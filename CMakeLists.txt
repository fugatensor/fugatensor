cmake_minimum_required(VERSION 3.28)
project(FugaTensor VERSION 0.1.0 LANGUAGES CXX)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE FUGA_NANOBIND_CMAKE_DIR
  RESULT_VARIABLE FUGA_NANOBIND_STATUS
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT FUGA_NANOBIND_STATUS EQUAL 0)
  message(FATAL_ERROR "FugaTensor requires nanobind 3.1.0 in the selected build environment")
endif()
find_package(nanobind 3.1.0 EXACT CONFIG REQUIRED PATHS "${FUGA_NANOBIND_CMAKE_DIR}" NO_DEFAULT_PATH)

# This core handles SSA dependency and liveness only. Tensor execution remains
# in ASK, and building this extension needs no LLVM, CUDA or DSL compiler SDK.
nanobind_add_module(_native NB_STATIC cpp/bindings.cpp cpp/graph.cpp)
target_compile_features(_native PRIVATE cxx_std_17)
target_include_directories(_native PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
target_compile_definitions(_native PRIVATE FUGA_VERSION="${PROJECT_VERSION}")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(_native PRIVATE -Wall -Wextra)
endif()
install(TARGETS _native LIBRARY DESTINATION fugatensor RUNTIME DESTINATION fugatensor)
