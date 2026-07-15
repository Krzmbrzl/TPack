# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.15...4.0)

project(tpack_example LANGUAGES CXX)

add_executable(tpack_example example.cpp)

find_package(TPack REQUIRED)

target_link_libraries(tpack_example PRIVATE TPack::TPack)
