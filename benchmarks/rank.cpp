// SPDX-License-Identifier: BSD-3-Clause

#include <tpack/orbit.hpp>
#include <tpack/rank.hpp>

#include <cstdint>
#include <random>
#include <vector>

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


static void BM_unrank_with_access(benchmark::State &state) {
	const std::vector< std::vector< std::vector< std::size_t > > > partitions = { { { 0, 1 }, { 2, 3 } } };
	const std::vector< std::size_t > dimensions                               = { 300, 300, 30, 30 };

	std::vector< double > dummy(300 * 300 * 30 * 30);
	std::random_device rnd_device;
	std::mt19937_64 engine(rnd_device());
	std::uniform_real_distribution< double > dist(0, 1);
	for (double &val : dummy) {
		val = dist(engine);
	}

	auto flat_idx = [dimensions](const auto &indexing) {
		std::size_t idx    = 0;
		std::size_t stride = 1;
		for (std::size_t i = 0; i < indexing.size(); ++i) {
			idx += indexing[i] * stride;
			stride *= dimensions[i];
		}
		return idx;
	};

	std::size_t num_orbits = tpack::num_orbits(dimensions, partitions);

	for (auto _ : state) {
		for (std::size_t rank = 0; rank < num_orbits; ++rank) {
			const auto indexing = tpack::unrank(12, dimensions, partitions);
			double element      = dummy.at(flat_idx(indexing));
			benchmark::DoNotOptimize(element);
		}
	}
}

BENCHMARK(BM_unrank_with_access);
