# CMake 构建（**备选路径，暂未验证**；日常用 setup.py 就够了，这条先放着）
#
#   cmake -S . -B build_cmake -DPython_EXECUTABLE=D:/ProgramData/Anaconda3/envs/dog/python.exe
#   cmake --build build_cmake --config Release
#
# 产物在 build_cmake/Release/_minitorch.cp310-win_amd64.pyd，
# 拷到仓库根目录（或装进环境）就能 import。
cmake_minimum_required(VERSION 3.18)
project(minitorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 指定 conda 环境里的解释器，pybind11 的头文件才会跟着那个环境走
if (NOT DEFINED Python_EXECUTABLE)
    set(Python_EXECUTABLE "D:/ProgramData/Anaconda3/envs/dog/python.exe" CACHE FILEPATH "Python interpreter")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# 库源码：tensor/ nn/ optimizer/ 全都要编进扩展
file(GLOB_RECURSE MT_LIB_SRC CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/tensor/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nn/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/optimizer/*.cpp")

pybind11_add_module(_minitorch
    "${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp"
    ${MT_LIB_SRC})

target_include_directories(_minitorch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
if (MSVC)
    target_compile_options(_minitorch PRIVATE /utf-8 /bigobj /W3)
endif()

# 纯 C++ 的演示程序（和 pybind11 无关，方便用 CMake 一起构建）
add_executable(miniptorch_demo
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    ${MT_LIB_SRC})
target_include_directories(miniptorch_demo PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
if (MSVC)
    target_compile_options(miniptorch_demo PRIVATE /utf-8 /bigobj /W3)
endif()
