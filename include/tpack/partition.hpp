// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/details/level_columns_view.hpp>

#include <algorithm>
#include <functional>
#include <ranges>
#include <type_traits>

namespace tpack {

namespace details {

	/**
	 * @brief Brings a partition specification into a canonical, layout-dependent order.
	 *
	 * Normalises three nested orderings so that two specifications describing the same symmetry produce the
	 * same rank()/unrank() layout: indices within each level, levels within each partition (ordered so the
	 * level holding the partition's extremal index comes first), and the partitions among themselves. For the
	 * column-major variant the smallest index is treated as extremal (sorted ascending); for row-major it is
	 * the largest. The index-to-dimension mapping is unaffected, so the reordering is purely organisational.
	 *
	 * @tparam col_major @c true to order for a column-major (smallest index first) layout, @c false for
	 *         row-major (largest index first).
	 * @tparam Partitions A range of partitions, each a range of levels, each level a range of index positions.
	 * @param partitions [in,out] The partition specification to reorder in place.
	 */
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

/**
 * @brief Canonicalises a partition specification for a column-major index layout.
 *
 * Reorders indices, levels and partitions in place so that the smallest index of each group is treated as
 * extremal (sorted first). Specifications that differ only in the order of equivalent indices, levels or
 * partitions become identical, yielding a stable rank()/unrank() layout.
 *
 * @tparam Partitions A range of partitions, each a range of levels, each level a range of index positions.
 * @param partitions [in,out] The partition specification to reorder in place.
 *
 * @see sort_partition_row_major
 */
template< std::ranges::range Partitions > void sort_partition_col_major(Partitions &&partitions) {
	details::sort_partition< true >(partitions);
}

/**
 * @brief Canonicalises a partition specification for a row-major index layout.
 *
 * Like sort_partition_col_major but treats the largest index of each group as extremal (sorted first),
 * matching a row-major storage order.
 *
 * @tparam Partitions A range of partitions, each a range of levels, each level a range of index positions.
 * @param partitions [in,out] The partition specification to reorder in place.
 *
 * @see sort_partition_col_major
 */
template< std::ranges::range Partitions > void sort_partition_row_major(Partitions &&partitions) {
	details::sort_partition< false >(partitions);
}

} // namespace tpack
