cmake_minimum_required(VERSION 3.24...4.3)

project(TensorPacking
	LANGUAGES CXX
	VERSION 1.0.0
)

add_library(tensor_packing INTERFACE)
target_compile_features(tensor_packing INTERFACE cxx_std_20)

target_sources(tensor_packing
	PUBLIC FILE_SET tpack_header TYPE HEADERS BASE_DIRS "${PROJECT_SOURCE_DIR}/include"
	FILES
		"include/tpack/details/binomial.hpp"
		"include/tpack/details/factorial.hpp"
		"include/tpack/details/level_columns_view.hpp"
		"include/tpack/orbit.hpp"
		"include/tpack/partition.hpp"
		"include/tpack/rank.hpp"
)

add_library(TPack::TPack ALIAS tensor_packing)
set_target_properties(tensor_packing PROPERTIES EXPORT_NAME "TPack")

if (PROJECT_IS_TOP_LEVEL)
	include(CheckCXXCompilerFlag)

	check_cxx_compiler_flag("-Wall;-Wextra;-Wpedantic" SUPPORTS_WARNING_FLAGS)
	check_cxx_compiler_flag("-Werror" SUPPORTS_WAE_FLAG)

	if (SUPPORTS_WARNING_FLAGS)
		target_compile_options(tensor_packing INTERFACE "$<BUILD_INTERFACE:-Wall>" "$<BUILD_INTERFACE:-Wextra>" "$<BUILD_INTERFACE:-Wpedantic>")
	endif()
	if (SUPPORTS_WAE_FLAG)
		target_compile_options(tensor_packing INTERFACE "$<BUILD_INTERFACE:-Werror>")
	endif()

	set(ENABLE_BENCHMARKS_DEFAULT ON)
else()
	set(ENABLE_BENCHMARKS_DEFAULT OFF)
endif()

option(TPACK_ENABLE_TESTING "Whether to build tests" ON)
option(TPACK_ENABLE_BENCHMARKS "Whether to build benchmarks" ${ENABLE_BENCHMARKS_DEFAULT})
option(TPACK_INSTALL "Whether to configure an install target for TPack" ON)

if (TPACK_ENABLE_TESTING)
	include(CTest)
	add_subdirectory(tests)
endif()

if (TPACK_ENABLE_BENCHMARKS)
	add_subdirectory(benchmarks)
endif()


export(TARGETS tensor_packing NAMESPACE TPack:: FILE TPackTargets.cmake)

include(CMakePackageConfigHelpers)
configure_package_config_file("tpack-config.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/tpack-config.cmake"
	INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tpack-${PROJECT_VERSION}"
)
write_basic_package_version_file(tpack-config-version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

if (TPACK_INSTALL)
	include(GNUInstallDirs)

	install(
		TARGETS tensor_packing
		EXPORT TPackTargets
		FILE_SET tpack_header DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
		  COMPONENT tpack_dev
	)

	# Generate and install a targets file (basically the same we created via export(…) in the
	# build tree but for the install tree
	install(
		EXPORT TPackTargets
		DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tpack-${PROJECT_VERSION}"
		FILE TPackTargets.cmake
		NAMESPACE TPack::
		COMPONENT tpack_dev
	)

	install(
	  FILES
		"${CMAKE_CURRENT_BINARY_DIR}/tpack-config.cmake"
		"${CMAKE_CURRENT_BINARY_DIR}/tpack-config-version.cmake"
		DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tpack-${PROJECT_VERSION}"
	)
endif()
