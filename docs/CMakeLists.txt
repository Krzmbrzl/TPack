# SPDX-License-Identifier: BSD-3-Clause

# Extract the tagged code snippets from the Markdown documentation and build (and,
# where requested, run) them so the examples in the docs keep matching the API.

find_package(Python3 COMPONENTS Interpreter QUIET)
if (NOT Python3_FOUND)
	message(WARNING "Python3 not found; documentation code snippets will not be built or tested.")
	return()
endif()

file(GLOB _tpack_doc_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.md")

set(_tpack_snippet_script "${CMAKE_CURRENT_SOURCE_DIR}/extract_snippets.py")
set(_tpack_snippet_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(_tpack_snippet_manifest "${_tpack_snippet_dir}/snippets.cmake")

# Re-run CMake (and hence the extraction below) whenever the docs or the
# extractor change.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
	${_tpack_doc_sources} "${_tpack_snippet_script}")

execute_process(
	COMMAND "${Python3_EXECUTABLE}" -I "${_tpack_snippet_script}"
		--docs-dir "${CMAKE_CURRENT_SOURCE_DIR}"
		--out-dir "${_tpack_snippet_dir}"
		--manifest "${_tpack_snippet_manifest}"
	RESULT_VARIABLE _tpack_snippet_rc
	OUTPUT_VARIABLE _tpack_snippet_out
	ERROR_VARIABLE _tpack_snippet_err
)
if (NOT _tpack_snippet_rc EQUAL 0)
	message(FATAL_ERROR "Failed to extract documentation snippets:\n${_tpack_snippet_out}${_tpack_snippet_err}")
endif()

include("${_tpack_snippet_manifest}")

foreach (_name IN LISTS TPACK_DOC_SNIPPETS)
	set(_tgt "tpack_doc_snippet_${_name}")

	# Compiled as a standalone consumer (include path only, no library build flags)
	# so the snippets exercise the public headers exactly as a downstream user would.
	add_executable(${_tgt} "${_tpack_snippet_dir}/${_name}.cpp")
	target_link_libraries(${_tgt} PRIVATE TPack::TPack)
	set_target_properties(${_tgt} PROPERTIES CXX_SCAN_FOR_MODULES OFF)

	if (TPACK_DOC_SNIPPET_${_name}_RUN)
		add_test(NAME "TPack.DocSnippet.${_name}" COMMAND ${_tgt})
	endif()
endforeach()
