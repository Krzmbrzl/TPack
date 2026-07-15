include(FetchContent)

FetchContent_Declare(
	googletest
	GIT_REPOSITORY "https://github.com/google/googletest"
	GIT_TAG        "v1.17.0"
	GIT_SHALLOW    ON
	EXCLUDE_FROM_ALL
	SYSTEM
	FIND_PACKAGE_ARGS NAMES gtest googletest
)

FetchContent_MakeAvailable(googletest)

add_executable(tensor_packing_tests
	"binomial.cpp"
	"column_view.cpp"
	"orbit.cpp"
	"partition.cpp"
	"rank.cpp"
)
target_link_libraries(tensor_packing_tests PRIVATE tensor_packing gtest_main)
set_target_properties(tensor_packing_tests PROPERTIES CXX_SCAN_FOR_MODULES OFF)


include(GoogleTest)
gtest_discover_tests(tensor_packing_tests)
