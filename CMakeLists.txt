cmake_minimum_required(VERSION 3.23)
# cmake_policy(SET CMP0104 NEW)

project(torch_radon LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

find_package(
  Python
  COMPONENTS Development
  REQUIRED)

find_package(PythonExtensions REQUIRED) # Function from scikit-build

set(CMAKE_PREFIX_PATH
    "${CMAKE_PREFIX_PATH};${PYTHON_SITE_PACKAGES_DIR}/torch/share/cmake")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(TR_CXX_DIR "${CMAKE_SOURCE_DIR}/src")
add_library(
  torch_radon_cuda SHARED
  "${TR_CXX_DIR}/backprojection.cu"
  "${TR_CXX_DIR}/fft.cu"
  "${TR_CXX_DIR}/forward.cu"
  "${TR_CXX_DIR}/log.cpp"
  "${TR_CXX_DIR}/noise.cu"
  "${TR_CXX_DIR}/parameter_classes.cu"
  "${TR_CXX_DIR}/pytorch.cpp"
  "${TR_CXX_DIR}/symbolic.cpp"
  "${TR_CXX_DIR}/texture.cu")

# python_extension_module(torch_radon_cuda)

set_target_properties(
  torch_radon_cuda
  PROPERTIES PREFIX "" CUDA_SEPARABLE_COMPILATION ON
             POSITION_INDEPENDENT_CODE ON CXX_STANDARD 14)

target_include_directories(
  torch_radon_cuda
  PUBLIC "${CMAKE_SOURCE_DIR}/include"
  PRIVATE "${TORCH_INCLUDE_DIRS}" "${Python_INCLUDE_DIRS}"
          "${CUDAToolkit_INCLUDE_DIRS}")

target_link_libraries(torch_radon_cuda ${TORCH_LIBRARIES})

install(TARGETS torch_radon_cuda LIBRARY DESTINATION "src/python")

# The following code block is suggested to be used on Windows. According to
# https://github.com/pytorch/pytorch/issues/25457, the DLLs need to be copied to
# avoid memory errors.
if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(
    TARGET example-app
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS}
            $<TARGET_FILE_DIR:example-app>)
endif(MSVC)
