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

# The installed C++ facade hides LLVM and owns the execution runtime.
find_package(ASK 0.1.0 EXACT CONFIG REQUIRED COMPONENTS Compiler Runtime)
nanobind_add_module(_native NB_STATIC cpp/bindings.cpp cpp/graph.cpp cpp/runtime.cpp
  cpp/compiler.cpp cpp/ask_runtime.cpp)
target_link_libraries(_native PRIVATE ASK::Compiler ASK::Runtime)
set_target_properties(_native PROPERTIES INSTALL_RPATH "$ORIGIN/_lib")
foreach(component Compiler Runtime)
  install(FILES $<TARGET_FILE:ASK::${component}> DESTINATION fugatensor/_lib
    RENAME $<TARGET_SONAME_FILE_NAME:ASK::${component}>)
endforeach()
target_compile_features(_native PRIVATE cxx_std_17)
target_include_directories(_native PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
target_compile_definitions(_native PRIVATE FUGA_VERSION="${PROJECT_VERSION}")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(_native PRIVATE -Wall -Wextra)
endif()
install(TARGETS _native LIBRARY DESTINATION fugatensor RUNTIME DESTINATION fugatensor)
