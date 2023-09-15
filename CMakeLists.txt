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


# refer to https://github.com/v6d-io/v6d/blob/main/cmake/CheckGCCABI.cmake
macro(check_gcc_cxx11abi)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -v
            OUTPUT_VARIABLE GCC_VERSION_OUT
            ERROR_VARIABLE GCC_VERSION_ERR)
    if(GCC_VERSION_OUT MATCHES ".*with-default-libstdcxx-abi=new.*" OR GCC_VERSION_ERR MATCHES ".*with-default-libstdcxx-abi=new.*")
      set(GCC_USE_CXX11_ABI 1)
    else()
      if(GCC_VERSION_OUT MATCHES ".*gcc4-compatible.*" OR GCC_VERSION_ERR MATCHES ".*gcc4-compatible.*")
        set(GCC_USE_CXX11_ABI 0)
      else()
        set(GCC_USE_CXX11_ABI 1)
      endif()
    endif()
  else()
    set(GCC_USE_CXX11_ABI 1)
  endif()
endmacro(check_gcc_cxx11abi)

check_gcc_cxx11abi()
message("GCC_USE_CXX11_ABI: ${GCC_USE_CXX11_ABI}")

# for glt_vineyard, set GCC_USE_CXX11_ABI_FLAG according to gcc environment
set(GCC_USE_CXX11_ABI_FLAG "_GLIBCXX_USE_CXX11_ABI=${GCC_USE_CXX11_ABI}")

# for glt, always use false (same as pytorch)
set(GCC_USE_CXX11_ABI_FALSE_FLAG "_GLIBCXX_USE_CXX11_ABI=0")

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
  -Werror=return-type)

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

function(target_source_tree target)
  file(GLOB_RECURSE SRCS ${ARGN})
  target_sources(${target} PRIVATE ${SRCS})
endfunction()

# Build graphlearn_torch_vineyard library
function(add_library_graphlearn_torch_vineyard)
  set(GLT_V6D_CSRC_DIR ${GLT_ROOT}/graphlearn_torch/v6d)
  find_package(vineyard REQUIRED)
  add_library(graphlearn_torch_vineyard SHARED)
  target_source_tree(graphlearn_torch_vineyard 
    ${GLT_V6D_CSRC_DIR}/*.cc
    PROPERTIES COMPILE_FLAGS ${GLT_CXX_FLAGS})
  target_include_directories(graphlearn_torch_vineyard
    PUBLIC ${GLT_ROOT})
  target_include_directories(graphlearn_torch_vineyard
    PRIVATE ${VINEYARD_INCLUDE_DIRS})

  target_link_libraries(graphlearn_torch_vineyard
    PUBLIC Python3::Python ${TORCH_LIBRARIES})
  target_link_libraries(graphlearn_torch_vineyard
    PUBLIC ${VINEYARD_LIBRARIES})
  target_compile_definitions(graphlearn_torch_vineyard PRIVATE GCC_USE_CXX11_ABI_FLAG)
endfunction()

if(WITH_VINEYARD)
  add_library_graphlearn_torch_vineyard()
endif()

function(add_library_graphlearn_torch)
  add_library(graphlearn_torch SHARED)
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
  message("TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
  target_link_libraries(graphlearn_torch
    PUBLIC Python3::Python ${TORCH_LIBRARIES})
  target_compile_definitions(graphlearn_torch PRIVATE GCC_USE_CXX11_ABI_FALSE_FLAG)

  if(WITH_CUDA)
    target_set_cuda_options(graphlearn_torch)
  endif()
endfunction()

add_library_graphlearn_torch()

# Build tests
if(BUILD_TESTS)
  set (GTest_INSTALL_DIR ${GLT_THIRD_PARTY_DIR}/googletest/build)
  find_package(GTest REQUIRED PATHS ${GTest_INSTALL_DIR})

  function(add_tests target library abi_flag)
    add_executable(${target} ${ARGN})
    add_dependencies(${target} ${library})
    target_link_libraries(${target}
      PUBLIC ${library} GTest::gtest GTest::gtest_main)
    if(WITH_CUDA)
      target_set_cuda_options(${target})
    endif()
    target_compile_definitions(${target} PRIVATE ${abi_flag})
  endfunction()

  if(WITH_CUDA)
    file(GLOB GLT_TEST_FILES ${GLT_CTEST_DIR}/test_*.cu)
  endif()

  foreach(t ${GLT_TEST_FILES})
    get_filename_component(name ${t} NAME_WE)
    add_tests(${name} graphlearn_torch GCC_USE_CXX11_ABI_FALSE_FLAG ${t})
  endforeach()
  if (WITH_VINEYARD)
    file(GLOB GLT_V6D_TEST_FILES ${GLT_CTEST_DIR}/test_vineyard.cc)
    foreach(t ${GLT_V6D_TEST_FILES})
      get_filename_component(name ${t} NAME_WE)
      add_tests(${name} graphlearn_torch_vineyard GCC_USE_CXX11_ABI_FLAG ${t})
    endforeach()
  endif()
endif()
