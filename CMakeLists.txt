cmake_minimum_required(VERSION 3.24...4.3 REQUIRED)

project(TensorPacking
	LANGUAGES CXX
)

add_library(tensor_packing INTERFACE)
target_compile_features(tensor_packing INTERFACE cxx_std_20)

if (PROJECT_IS_TOP_LEVEL)
	include(CheckCXXCompilerFlag)

	check_cxx_compiler_flag("-Wall;-Wextra;-Wpedantic" SUPPORTS_WARNING_FLAGS)
	check_cxx_compiler_flag("-Werror" SUPPORTS_WAE_FLAG)

	if (SUPPORTS_WARNING_FLAGS)
		target_compile_options(tensor_packing INTERFACE "-Wall" "-Wextra" "-Wpedantic")
	endif()
	if (SUPPORTS_WAE_FLAG)
		target_compile_options(tensor_packing INTERFACE "-Werror")
	endif()
else()
	set(INCLUDE_AS_SYSTEM "SYSTEM")
endif()

target_include_directories(tensor_packing ${INCLUDE_AS_SYSTEM} INTERFACE "${PROJECT_SOURCE_DIR}/include")

option(TPACK_ENABLE_TESTING "Whether to build tests" ON)

if (TPACK_ENABLE_TESTING)
	include(CTest)
	add_subdirectory(tests)
endif()
