cmake_minimum_required(VERSION 3.10)
project(GraphLearn-for-PyTorch)
set(CMAKE_CXX_STANDARD 14)

option(DEBUG "Enable debug mode" OFF)
option(BUILD_TESTS "Enable testing" ON)
option(WITH_CUDA "Enable CUDA support" ON)
option(WITH_VINEYARD "Enable vineyard support" OFF)

set(GLT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(GLT_CSRC_DIR ${GLT_ROOT}/graphlearn_torch/csrc)
set(GLT_CTEST_DIR ${GLT_ROOT}/test/cpp)
set(GLT_BUILT_DIR ${GLT_ROOT}/built)
set(GLT_THIRD_PARTY_DIR ${GLT_ROOT}/third_party)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLT_BUILT_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${GLT_BUILT_DIR}/lib)

if(DEBUG)
  set(GLT_MODE_FLAGS -g)
  set(CMAKE_BUILD_TYPE Debug)
else()
  set(GLT_MODE_FLAGS -O2)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(GLT_CXX_FLAGS
  ${GLT_MODE_FLAGS}
  -fPIC
  -fvisibility-inlines-hidden
  -mavx
  -msse4.2
  -msse4.1
  -Wno-attributes
  -Wno-deprecated-declarations
  -Werror=return-type
  -D_GLIBCXX_USE_CXX11_ABI=0)

# Link to Python when building
find_package(PythonInterp REQUIRED)
find_package(Python3 COMPONENTS Development)

# Link to CUDA if enabled
if(WITH_CUDA)
  find_package(CUDA REQUIRED)
  enable_language(CUDA)
  add_definitions(-DHAVE_CUDA=1)

  # Auto-detect CUDA architectures.
  include(FindCUDA/select_compute_arch)
  CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
  string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
  string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
  string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
  set(TORCH_CUDA_ARCH_LIST ${INSTALLED_GPU_CCS_3})
  set(CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

  function(target_set_cuda_options target)
    set_property(
      TARGET ${target}
      PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
    target_include_directories(${target}
      PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/include)
  endfunction()
endif()

## protobuf and gRPC package
set (gRPC_INSTALL_DIR ${GLT_THIRD_PARTY_DIR}/grpc/build)
list (APPEND CMAKE_PREFIX_PATH ${gRPC_INSTALL_DIR})
find_package (protobuf CONFIG REQUIRED)

macro (get_target_import_location var target)
  if (TARGET ${target})
    foreach (prop IMPORTED_LOCATION IMPORTED_LOCATION_NOCONFIG IMPORTED_LOCATION_DEBUG IMPORTED_LOCATION_RELEASE)
      get_target_property (${var} ${target} ${prop})
      if (NOT ("${${var}}" STREQUAL "${var}-NOTFOUND"))
        break ()
      endif ()
    endforeach ()
  endif ()
endmacro ()

find_package (gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC: gRPC_found=${gRPC_FOUND}")
if (gRPC_FOUND)
  if (TARGET gRPC::grpc_cpp_plugin)
    get_target_import_location(GLT_GRPC_CPP_PLUGIN gRPC::grpc_cpp_plugin)
  else ()
    set (GLT_GRPC_CPP_PLUGIN ${gRPC_INSTALL_DIR}/bin/grpc_cpp_plugin)
  endif ()
else ()
  include (${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindGRPC.cmake)
  set_target_properties(gRPC::grpc++ PROPERTIES INTERFACE_LINK_LIBRARIES "${Protobuf_LIBRARIES}")
  get_target_import_location(GLT_GRPC_CPP_PLUGIN gRPC::grpc_cpp_plugin)
endif ()

# message(STATUS "Using gRPC CPP plugin: ${GLT_GRPC_CPP_PLUGIN}")

## grpc codegen
set(RPC_SRC_DIR ${GLT_ROOT}/graphlearn_torch/csrc/rpc)
set (GLT_PROTO_GEN_DIR ${RPC_SRC_DIR}/generated)
set (GLT_PROTO_DIR ${RPC_SRC_DIR}/proto)
file (MAKE_DIRECTORY ${GLT_PROTO_GEN_DIR})
set (GLT_PROTO_SRC_FILES
  ${GLT_PROTO_DIR}/helloworld.proto)

function (generate_grpc_files target_name grpc_gen_files)
  set (${grpc_gen_files})
  foreach (FIL ${GLT_PROTO_SRC_FILES})
    get_filename_component (FIL_WE ${FIL} NAME_WE)
    set (GEN_FILES
      ${GLT_PROTO_GEN_DIR}/${FIL_WE}.pb.h
      ${GLT_PROTO_GEN_DIR}/${FIL_WE}.pb.cc
      ${GLT_PROTO_GEN_DIR}/${FIL_WE}.grpc.pb.h
      ${GLT_PROTO_GEN_DIR}/${FIL_WE}.grpc.pb.cc)
    list (APPEND ${grpc_gen_files} ${GEN_FILES})
    add_custom_command (
      OUTPUT ${GEN_FILES}
      DEPENDS protobuf::protoc gRPC::grpc_cpp_plugin ${FIL}
      COMMENT "Generating for proto file: ${FIL_WE}.proto ..."
      COMMAND protobuf::protoc -I ${GLT_PROTO_DIR} --cpp_out=${GLT_PROTO_GEN_DIR} ${FIL}
      COMMAND protobuf::protoc -I ${GLT_PROTO_DIR} --grpc_out=${GLT_PROTO_GEN_DIR}
        --plugin=protoc-gen-grpc=${GLT_GRPC_CPP_PLUGIN} ${FIL}
      VERBATIM)
  endforeach ()

  set_source_files_properties (${${grpc_gen_files}} PROPERTIES GENERATED TRUE)
  set (${grpc_gen_files} ${${grpc_gen_files}} PARENT_SCOPE)

  add_custom_target (${target_name}
    DEPENDS ${${grpc_gen_files}})
endfunction ()

generate_grpc_files (grpc_codegen GLT_GRPC_AUTOGEN_FILES)

# Link to PyTorch
# Auto-find CMAKE_PREFIX_PATH for PyTorch
execute_process(COMMAND
  python -c "import torch;print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_DIR
)
string(REGEX MATCH ".*/torch/" TORCH_DIR ${TORCH_DIR})
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
find_package(Torch REQUIRED)

if(WITH_CUDA)
  add_definitions(-DWITH_CUDA)
endif()

# Link to Vineyard if enabled
if(WITH_VINEYARD)
  add_definitions(-DWITH_VINEYARD)
  find_package(vineyard REQUIRED)
endif()

# Build library
add_library(graphlearn_torch SHARED
  ${GLT_GRPC_AUTOGEN_FILES} )

add_dependencies(graphlearn_torch grpc_codegen)


function(target_source_tree target)
  file(GLOB_RECURSE SRCS ${ARGN})
  target_sources(${target} PRIVATE ${SRCS})
endfunction()

target_source_tree(graphlearn_torch
  ${GLT_CSRC_DIR}/*.cc
  PROPERTIES COMPILE_FLAGS ${GLT_CXX_FLAGS})
target_source_tree(graphlearn_torch
  ${GLT_CSRC_DIR}/cpu/*.cc
  PROPERTIES COMPILE_FLAGS ${GLT_CXX_FLAGS})
if(WITH_CUDA)
  target_source_tree(graphlearn_torch
    ${GLT_CSRC_DIR}/cuda/*.cu)
endif()

target_include_directories(graphlearn_torch
  PUBLIC ${GLT_ROOT})

target_link_libraries(graphlearn_torch
  PUBLIC 
    Python3::Python 
    ${TORCH_LIBRARIES} 
    gRPC::grpc++ 
    gRPC::grpc++_reflection
)

if(WITH_CUDA)
  target_set_cuda_options(graphlearn_torch)
endif()

if(WITH_VINEYARD)
  target_include_directories(graphlearn_torch
    PRIVATE ${VINEYARD_INCLUDE_DIRS})
  target_link_libraries(graphlearn_torch
    PUBLIC ${VINEYARD_LIBRARIES})
endif()


# Build tests
if(BUILD_TESTS)
  set (GTest_INSTALL_DIR ${GLT_THIRD_PARTY_DIR}/googletest/build)
  find_package(GTest REQUIRED PATHS ${GTest_INSTALL_DIR})

  function(glt_add_test target)
    add_executable(${target} ${ARGN})
    add_dependencies(${target} graphlearn_torch)
    target_link_libraries(${target}
      PUBLIC graphlearn_torch GTest::gtest GTest::gtest_main)
    if(WITH_CUDA)
      target_set_cuda_options(${target})
    endif()
  endfunction()

  if(WITH_CUDA)
    file(GLOB GLT_TEST_FILES ${GLT_CTEST_DIR}/test_*.cu)
  endif()

  message(STATUS "GLT_TEST_FILES: ${GLT_TEST_FILES}")
  foreach(t ${GLT_TEST_FILES})
    get_filename_component(name ${t} NAME_WE)
    glt_add_test(${name} ${t})
  endforeach()
endif()
