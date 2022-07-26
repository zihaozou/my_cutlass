cmake_minimum_required(VERSION 3.18)
PROJECT(
	my_cutlass
	LANGUAGES CXX CUDA
)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No release type specified. Setting to 'Release'.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
###############################################################################
# CUDA compiler setup
###############################################################################

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
list(FILTER CMAKE_CUDA_ARCHITECTURES EXCLUDE REGEX "PTX")
if (DEFINED ENV{TCNN_CUDA_ARCHITECTURES})
	message(STATUS "Obtained target architecture from environment variable TCNN_CUDA_ARCHITECTURES=$ENV{TCNN_CUDA_ARCHITECTURES}")
	set(CMAKE_CUDA_ARCHITECTURES $ENV{TCNN_CUDA_ARCHITECTURES})
endif()

if ((80 IN_LIST CMAKE_CUDA_ARCHITECTURES OR 86 IN_LIST CMAKE_CUDA_ARCHITECTURES) AND CUDA_VERSION VERSION_LESS 11.0)
	message(WARNING "CUDA version ${CUDA_VERSION} is too low for targeting Ampere GPUs. Reverting to compute capability 75.")
	list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES 80 86)
	if (NOT CMAKE_CUDA_ARCHITECTURES)
		list(APPEND CMAKE_CUDA_ARCHITECTURES 75)
	endif()
endif()
list(SORT CMAKE_CUDA_ARCHITECTURES COMPARE NATURAL ORDER ASCENDING)
list(GET CMAKE_CUDA_ARCHITECTURES 0 MIN_GPU_ARCH)

list(APPEND MY_CUTLASS_DEFINITIONS -DMY_CUTLASS_MIN_GPU_ARCH=${MIN_GPU_ARCH})


message(STATUS "Targeting GPU architectures: ${CMAKE_CUDA_ARCHITECTURES}")

if (MSVC)
else()
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-mf16c")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
	list(APPEND CUDA_NVCC_FLAGS "-Xcudafe=--diag_suppress=unrecognized_gcc_pragma")
endif()
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
find_package(benchmark REQUIRED)



include_directories("dependencies/cutlass/include")
include_directories("dependencies/cutlass/tools/util/include")
include_directories("dependencies/cutlass/examples/common")
set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library" FORCE)
add_subdirectory("dependencies/cutlass")
include_directories("dependencies/patterns/include")
add_subdirectory("dependencies/patterns")

include_directories("includes")
set(MY_CUTLASS_SOURCES sources/cutlass_conv.cu)

add_library(my_cutlass STATIC ${MY_CUTLASS_SOURCES})
target_link_libraries(my_cutlass PUBLIC mpark_patterns)
target_compile_definitions(my_cutlass PUBLIC ${MY_CUTLASS_DEFINITIONS})
target_compile_options(my_cutlass PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
# add_executable(main main.cu)
# target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

enable_testing()

add_executable(
  unit_tests
  tests/unit_test.cu
)
target_link_libraries(
  unit_tests
  GTest::gtest_main
  my_cutlass
)

include(GoogleTest)
gtest_discover_tests(unit_tests)

# add_executable(performance benchmarks/benchmark.cu)
# target_link_libraries(
#   performance
#   my_cutlass
#   benchmark::benchmark
# )
# target_link_libraries(main PUBLIC ${CUDA_LIBRARIES})