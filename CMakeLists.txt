cmake_minimum_required(VERSION 3.10)

project(MSD-radixsort)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)
include(default_build_type)

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -DNDEBUG -march=native") #  -fmax-errors=2
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -march=native") #  -fmax-errors=2

add_library(inpmsdradix STATIC src/msb_64.c src/rand.c)

target_include_directories(inpmsdradix PUBLIC include)
