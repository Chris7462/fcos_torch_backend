cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(fcos_object_detection)

find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(fcos_object_detection
  src/fcos_inference.cpp)

target_include_directories(fcos_object_detection
  PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS})

target_link_libraries(fcos_object_detection
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    -Wl,--no-as-needed TorchVision::TorchVision)

set_property(TARGET fcos_object_detection PROPERTY CXX_STANDARD 17)
