// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/details/level_columns_view.hpp>

#include <algorithm>
#include <functional>
#include <ranges>
#include <type_traits>

namespace tpack {

namespace details {

	template< bool col_major, std::ranges::range Partitions > void sort_partition(Partitions &&partitions) {
		using std::ranges::begin;

		using cmp_less    = std::conditional_t< col_major, std::less<>, std::greater<> >;
		using cmp_greater = std::conditional_t< col_major, std::greater<>, std::less<> >;

		for (auto &&part_levels : partitions) {
			// Sort levels such that the one containing the smallest (col-major) or largest (row-major) index
			// comes first
			std::ranges::sort(part_levels, cmp_less{},
							  [](const auto &level) { return *std::ranges::min_element(level, cmp_less{}); });

			// Bring level columns into descending order
			details::LevelColumnsView columns(part_levels);
			std::ranges::sort(columns, cmp_greater{});
		}

		// The first level of every partition contains the partition's extremal index
		std::ranges::sort(partitions, cmp_less{}, [](const auto &part_levels) {
			return *std::ranges::min_element(*begin(part_levels), cmp_less{});
		});
	}

} // namespace details

template< std::ranges::range Partitions > void sort_partition_col_major(Partitions &&partitions) {
	details::sort_partition< true >(partitions);
}

template< std::ranges::range Partitions > void sort_partition_row_major(Partitions &&partitions) {
	details::sort_partition< false >(partitions);
}

} // namespace tpack
