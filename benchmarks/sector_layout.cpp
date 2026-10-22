// SPDX-License-Identifier: BSD-3-Clause

#include <tpack/labeled.hpp>
#include <tpack/sector_layout.hpp>

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
