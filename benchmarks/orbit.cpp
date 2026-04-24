// SPDX-License-Identifier: BSD-3-Clause

#include <tpack/orbit.hpp>

#include <cstdint>
#include <vector>

#include <benchmark/benchmark.h>


static void BM_num_orbits(benchmark::State &state) {
	std::size_t counter = 0;
	std::vector< std::vector< std::vector< std::size_t > > > partitions;
	std::vector< std::size_t > dimensions;
	for (std::int64_t part = 0; part < state.range(0); ++part) {
		partitions.emplace_back();
		for (std::int64_t level = 0; level < state.range(1); ++level) {
			partitions.back().emplace_back();
			for (std::int64_t col = 0; col < state.range(2); ++col) {
				partitions.back().back().emplace_back(counter++);
				dimensions.emplace_back(4);
			}
		}
	}

	for (auto _ : state) {
		std::size_t num = tpack::num_orbits(dimensions, partitions);
		benchmark::DoNotOptimize(num);
	}
}

BENCHMARK(BM_num_orbits)->ArgsProduct({ { 1, 2, 3, 4 }, { 1, 2, 3, 4 }, { 1, 2, 3, 4 } });
