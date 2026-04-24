// SPDX-License-Identifier: BSD-3-Clause

#include <tpack/rank.hpp>

#include <cstdint>

#include <benchmark/benchmark.h>


static void BM_rank(benchmark::State &state) {
	std::vector< std::size_t > indexing;
	std::size_t counter = 0;
	std::vector< std::vector< std::vector< std::size_t > > > partitions;
	std::vector< std::size_t > dimensions;
	for (std::int64_t part = 0; part < state.range(0); ++part) {
		partitions.emplace_back();
		for (std::int64_t level = 0; level < state.range(1); ++level) {
			partitions.back().emplace_back();
			for (std::int64_t col = 0; col < state.range(2); ++col) {
				partitions.back().back().emplace_back(counter++);
				dimensions.emplace_back(20);
				indexing.emplace_back(12);
			}
		}
	}

	for (auto _ : state) {
		std::size_t rank = tpack::rank(indexing, dimensions, partitions);
		benchmark::DoNotOptimize(rank);
	}
}

BENCHMARK(BM_rank)->ArgsProduct({ { 1, 2, 3, 4 }, { 1, 2, 3, 4 }, { 1, 2, 3, 4 } });


static void BM_unrank(benchmark::State &state) {
	std::size_t counter = 0;
	std::vector< std::vector< std::vector< std::size_t > > > partitions;
	std::vector< std::size_t > dimensions;
	for (std::int64_t part = 0; part < state.range(0); ++part) {
		partitions.emplace_back();
		for (std::int64_t level = 0; level < state.range(1); ++level) {
			partitions.back().emplace_back();
			for (std::int64_t col = 0; col < state.range(2); ++col) {
				partitions.back().back().emplace_back(counter++);
				dimensions.emplace_back(20);
			}
		}
	}

	for (auto _ : state) {
		auto indexing = tpack::unrank(12, dimensions, partitions);
		benchmark::DoNotOptimize(indexing);
	}
}

BENCHMARK(BM_unrank)->ArgsProduct({ { 1, 2, 3, 4 }, { 1, 2, 3, 4 }, { 1, 2, 3, 4 } });
