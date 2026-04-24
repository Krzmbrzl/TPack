include(FetchContent)

FetchContent_Declare(
	googlebenchmark
	GIT_REPOSITORY "https://github.com/google/benchmark"
	GIT_TAG        "v1.9.5"
	GIT_SHALLOW    ON
	EXCLUDE_FROM_ALL
	SYSTEM
	FIND_PACKAGE_ARGS NAMES benchmark googlebenchmark
)

FetchContent_MakeAvailable(googlebenchmark)

add_executable(tensor_packing_benchmarks
	"combinatoric.cpp"
	"main.cpp"
	"orbit.cpp"
	"rank.cpp"
)

target_link_libraries(tensor_packing_benchmarks PRIVATE tensor_packing benchmark::benchmark)
