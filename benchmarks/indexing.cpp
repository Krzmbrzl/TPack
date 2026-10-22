// SPDX-License-Identifier: BSD-3-Clause

#include <tpack/indexing.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <benchmark/benchmark.h>


static void BM_canonicalize(benchmark::State &state) {
	std::size_t counter = 0;
	std::vector< std::vector< std::vector< std::size_t > > > partitions;
	std::vector< std::size_t > indexing;
	for (std::int64_t part = 0; part < state.range(0); ++part) {
		partitions.emplace_back();
		for (std::int64_t level = 0; level < state.range(1); ++level) {
			partitions.back().emplace_back();
			for (std::int64_t col = 0; col < state.range(2); ++col) {
				partitions.back().back().emplace_back(counter++);
				// Ascending columns are the worst case: canonicalize must fully reverse them.
				indexing.emplace_back(static_cast< std::size_t >(col));
			}
		}
	}

	std::vector< std::size_t > work(indexing.size());
	for (auto _ : state) {
		// Restore the non-canonical input each iteration so every call does the full sort.
		std::ranges::copy(indexing, work.begin());
		int phase = tpack::canonicalize(work, partitions);
		benchmark::DoNotOptimize(phase);
		benchmark::DoNotOptimize(work);
	}
}

BENCHMARK(BM_canonicalize)->ArgsProduct({ { 1, 2, 3, 4 }, { 1, 2, 3, 4 }, { 1, 2, 3, 4 } });
