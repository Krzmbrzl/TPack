// SPDX-License-Identifier: BSD-3-Clause

#include <tpack/labeled.hpp>
#include <tpack/sector_layout.hpp>
#include <tpack/sector_walker.hpp>
#include <tpack/symmetry.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include <benchmark/benchmark.h>


namespace {

// Two slots, each carrying num_labels labels with count local values apiece.
std::vector< std::map< int, std::size_t > > make_dims(std::int64_t num_labels, std::int64_t count) {
	std::map< int, std::size_t > slot;
	for (std::int64_t label = 0; label < num_labels; ++label) {
		slot[static_cast< int >(label)] = static_cast< std::size_t >(count);
	}
	return std::vector< std::map< int, std::size_t > >(2, slot);
}

const std::vector< std::vector< std::vector< std::size_t > > > kPartitions{ { { 0, 1 } } };

} // namespace


static void BM_sector_layout_construct(benchmark::State &state) {
	const auto dims = make_dims(state.range(0), state.range(1));

	for (auto _ : state) {
		tpack::SectorLayout< int > layout(dims, kPartitions, tpack::xor_constraint(0, 0));
		benchmark::DoNotOptimize(layout);
	}
}

BENCHMARK(BM_sector_layout_construct)->ArgsProduct({ { 2, 4, 8 }, { 2, 4 } });


static void BM_sector_layout_unrank(benchmark::State &state) {
	const auto dims = make_dims(state.range(0), state.range(1));
	const tpack::SectorLayout< int > layout(dims, kPartitions, tpack::xor_constraint(0, 0));

	std::vector< tpack::Labeled< int > > idx(2);
	for (auto _ : state) {
		for (std::size_t r = 0; r < layout.size(); ++r) {
			layout.unrank(r, idx);
			benchmark::DoNotOptimize(idx);
		}
	}
}

BENCHMARK(BM_sector_layout_unrank)->ArgsProduct({ { 2, 4, 8 }, { 2, 4 } });


namespace {

struct SweepConfig {
	std::vector< std::map< int, std::size_t > > dimensions;
	std::vector< std::vector< std::vector< std::size_t > > > partitions;
	tpack::Symmetry symmetry;
};

// Args: partitions, levels, columns, antisymmetric (0/1). Every slot carries two labels with two local values each.
SweepConfig make_sweep_config(const benchmark::State &state) {
	SweepConfig config{ {}, {}, state.range(3) ? tpack::Symmetry::Antisymm : tpack::Symmetry::Symm };

	std::size_t counter = 0;
	for (std::int64_t part = 0; part < state.range(0); ++part) {
		config.partitions.emplace_back();
		for (std::int64_t level = 0; level < state.range(1); ++level) {
			config.partitions.back().emplace_back();
			for (std::int64_t col = 0; col < state.range(2); ++col) {
				config.partitions.back().back().emplace_back(counter++);
				config.dimensions.push_back({ { 0, 2 }, { 1, 2 } });
			}
		}
	}

	return config;
}

} // namespace

// Runs through all ranks by calling unrank for every single one of them
static void BM_sector_layout_unrank_sweep(benchmark::State &state) {
	const SweepConfig config = make_sweep_config(state);
	const tpack::SectorLayout< int > layout(config.dimensions, config.partitions, tpack::xor_constraint(0, 0),
											config.symmetry);

	std::vector< tpack::Labeled< int > > idx(config.dimensions.size());
	for (auto _ : state) {
		for (std::size_t r = 0; r < layout.size(); ++r) {
			layout.unrank(r, idx);
			benchmark::DoNotOptimize(idx);
		}
	}

	state.SetItemsProcessed(state.iterations() * static_cast< std::int64_t >(layout.size()));
}

// Runs through all ranks by incrementally advancing a SectorWalker
static void BM_sector_walker_sweep(benchmark::State &state) {
	const SweepConfig config = make_sweep_config(state);
	const tpack::SectorLayout< int > layout(config.dimensions, config.partitions, tpack::xor_constraint(0, 0),
											config.symmetry);

	tpack::SectorWalker< int > walker(layout);
	for (auto _ : state) {
		// The walker wraps around to rank 0 after the last rank, so it is ready for the next iteration.
		do {
			// The indexing is only exposed as const, so make the compiler assume that its contents are read.
			benchmark::DoNotOptimize(walker.indexing().data());
			benchmark::ClobberMemory();
		} while (walker.next());
	}

	state.SetItemsProcessed(state.iterations() * static_cast< std::int64_t >(layout.size()));
}

// clang-format off
BENCHMARK(BM_sector_layout_unrank_sweep)->ArgsProduct({ { 1, 2 }, { 1, 2 }, { 1, 2, 3 }, { 0, 1 } });
BENCHMARK(BM_sector_walker_sweep)->ArgsProduct({ { 1, 2 }, { 1, 2 }, { 1, 2, 3 }, { 0, 1 } });
// clang-format on
