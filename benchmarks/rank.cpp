// SPDX-License-Identifier: BSD-3-Clause

#include <tpack/orbit.hpp>
#include <tpack/rank.hpp>
#include <tpack/rank_walker.hpp>
#include <tpack/symmetry.hpp>

#include <cstddef>
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
				dimensions.emplace_back(6);
				indexing.emplace_back(5);
			}
		}
	}

	for (auto _ : state) {
		std::size_t rank = tpack::rank(indexing, dimensions, partitions);
		benchmark::DoNotOptimize(rank);
	}
}

// clang-format off
BENCHMARK(BM_rank)->ArgsProduct({ { 1, 2, 3 }, { 1, 2, 3, 4 }, { 1, 2 } });
// clang-format on


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
				dimensions.emplace_back(6);
			}
		}
	}

	std::random_device rnd_device;
	std::mt19937_64 engine(rnd_device());
	std::uniform_int_distribution< std::size_t > dist(0, 12);
	// Keep the rank within range; unrank's precondition is rank < num_orbits, and smaller configurations
	// have fewer orbits than the draw's upper bound.
	const std::size_t rank = dist(engine) % tpack::num_orbits(dimensions, partitions);

	std::vector< std::size_t > indexing(dimensions.size());

	for (auto _ : state) {
		tpack::unrank(rank, indexing, dimensions, partitions);
		benchmark::DoNotOptimize(indexing);
	}
}

// clang-format off
BENCHMARK(BM_unrank)->ArgsProduct({ { 1, 2, 3 }, { 1, 2, 3, 4 }, { 1, 2, } });
// clang-format on


static void BM_unrank_with_access(benchmark::State &state) {
	const std::vector< std::vector< std::vector< std::size_t > > > partitions = { { { 0, 1 }, { 2, 3 } } };
	const std::vector< std::size_t > dimensions                               = { 215, 215, 13, 13 };

	std::vector< double > dummy(215 * 215 * 13 * 13);
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

	std::vector< std::size_t > indexing(4);

	std::size_t num_orbits = tpack::num_orbits(dimensions, partitions);

	for (auto _ : state) {
		for (std::size_t rank = 0; rank < num_orbits; ++rank) {
			tpack::unrank(rank, indexing, dimensions, partitions);
			double element = dummy.at(flat_idx(indexing));
			benchmark::DoNotOptimize(element);
		}
	}
}

BENCHMARK(BM_unrank_with_access);


namespace {

struct SweepConfig {
	std::vector< std::vector< std::vector< std::size_t > > > partitions;
	std::vector< std::size_t > dimensions;
	tpack::Symmetry symmetry;
};

// Args: partitions, levels, columns, antisymmetric (0/1)
SweepConfig make_sweep_config(const benchmark::State &state) {
	SweepConfig config{ {}, {}, state.range(3) ? tpack::Symmetry::Antisymm : tpack::Symmetry::Symm };

	std::size_t counter = 0;
	for (std::int64_t part = 0; part < state.range(0); ++part) {
		config.partitions.emplace_back();
		for (std::int64_t level = 0; level < state.range(1); ++level) {
			config.partitions.back().emplace_back();
			for (std::int64_t col = 0; col < state.range(2); ++col) {
				config.partitions.back().back().emplace_back(counter++);
				config.dimensions.emplace_back(4);
			}
		}
	}

	return config;
}

} // namespace

// Runs through all ranks by calling unrank for every single one of them
static void BM_unrank_sweep(benchmark::State &state) {
	const SweepConfig config = make_sweep_config(state);
	const std::size_t count  = tpack::num_orbits(config.dimensions, config.partitions, config.symmetry);

	std::vector< std::size_t > indexing(config.dimensions.size());

	for (auto _ : state) {
		for (std::size_t rank = 0; rank < count; ++rank) {
			tpack::unrank(rank, indexing, config.dimensions, config.partitions, config.symmetry);
			benchmark::DoNotOptimize(indexing);
		}
	}

	state.SetItemsProcessed(state.iterations() * static_cast< std::int64_t >(count));
}

// Runs through all ranks by incrementally advancing a RankWalker
static void BM_rank_walker_sweep(benchmark::State &state) {
	const SweepConfig config = make_sweep_config(state);
	const std::size_t count  = tpack::num_orbits(config.dimensions, config.partitions, config.symmetry);

	tpack::RankWalker walker(config.dimensions, config.partitions, config.symmetry);

	for (auto _ : state) {
		// The walker wraps around to rank 0 after the last rank, so it is ready for the next iteration.
		do {
			// The indexing is only exposed as const, so make the compiler assume that its contents are read.
			benchmark::DoNotOptimize(walker.indexing().data());
			benchmark::ClobberMemory();
		} while (walker.next());
	}

	state.SetItemsProcessed(state.iterations() * static_cast< std::int64_t >(count));
}

// clang-format off
BENCHMARK(BM_unrank_sweep)->ArgsProduct({ { 1, 2 }, { 1, 2 }, { 1, 2, 3 }, { 0, 1 } });
BENCHMARK(BM_rank_walker_sweep)->ArgsProduct({ { 1, 2 }, { 1, 2 }, { 1, 2, 3 }, { 0, 1 } });
// clang-format on
