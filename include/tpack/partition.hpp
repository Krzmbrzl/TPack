// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/concepts.hpp>
#include <tpack/details/apply_permutation.hpp>
#include <tpack/details/level_columns_view.hpp>
#include <tpack/details/symmetry.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <ranges>
#include <type_traits>
#include <vector>

namespace tpack {

namespace details {

	/**
	 * @brief Reorders a partition specification to match a storage layout, for I/O-friendly (un)packing.
	 *
	 * Normalises three nested orderings so that the order in which rank()/unrank() walk the packed storage
	 * follows the tensor's memory layout: indices within each level, levels within each partition (ordered so
	 * the level holding the partition's extremal index comes first), and the partitions among themselves. For
	 * the column-major variant the smallest index is treated as extremal (sorted ascending); for row-major it
	 * is the largest. The index-to-dimension mapping is unaffected, so the reordering is purely organisational.
	 *
	 * Because the reordering is deterministic, it also canonicalises: two specifications describing the same
	 * symmetry collapse to the same ordering and thus the same rank()/unrank() layout.
	 *
	 * When a per-partition sign array is supplied it is reordered in lockstep with the partitions, so each
	 * partition keeps its antisymmetry flag.
	 *
	 * @tparam col_major @c true to order for a column-major (smallest index first) layout, @c false for
	 *         row-major (largest index first).
	 * @tparam Partitions A range of partitions, each a range of levels, each level a range of index positions.
	 * @tparam Signs A single Symmetry (applied to every partition), a random-access range of per-partition
	 *         Symmetry flags, or the all_symmetric_t sentinel.
	 * @param partitions [in,out] The partition specification to reorder in place.
	 * @param signs [in,out] The per-partition sign array to reorder in lockstep; a single Symmetry and the
	 *        all_symmetric_t sentinel apply uniformly and leave nothing to reorder.
	 */
	template< bool col_major, std::ranges::range Partitions, partition_signs Signs = all_symmetric_t >
	void sort_partitions(Partitions &&partitions, Signs &&signs = all_symmetric_t{}) {
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

		// The first level of every partition contains the partition's extremal index. Reorder partitions via an
		// index permutation so a parallel sign array can follow the same moves; partitions act on disjoint
		// indices, so the extremal-index keys are unique and the ordering is deterministic.
		auto part_key = [](const auto &part_levels) {
			return *std::ranges::min_element(*begin(part_levels), cmp_less{});
		};

		std::vector< std::size_t > keys;
		for (auto &&part_levels : partitions) {
			keys.push_back(part_key(part_levels));
		}

		std::vector< std::size_t > order =
			details::sorted_index_permutation(keys.size(), cmp_less{}, [&keys](std::size_t i) { return keys[i]; });

		if constexpr (std::ranges::random_access_range< Signs >) {
			assert(static_cast< std::size_t >(std::ranges::distance(signs)) == keys.size());
			details::apply_permutation(order, partitions, signs);
		} else {
			// The sentinel and a single Symmetry apply to every partition, so there is nothing to reorder.
			details::apply_permutation(order, partitions);
		}
	}

} // namespace details

/**
 * @brief The storage layout a partition specification is ordered for.
 *
 * Determines which index of each group is treated as extremal (sorted first): the smallest for a
 * column-major layout, the largest for a row-major one.
 */
enum class IndexOrder { ColumnMajor, RowMajor };

/**
 * @brief Reorders a partition specification to match a tensor's storage layout, improving (un)packing I/O.
 *
 * The order in which indices, levels and partitions appear in a specification fixes the order in which
 * rank()/unrank() traverse the packed storage. This reorders the specification in place so that the extremal
 * index of each group (the smallest for @c IndexOrder::ColumnMajor, the largest for @c IndexOrder::RowMajor) is
 * sorted first, aligning that traversal with the tensor's memory layout so bulk (un)packing visits storage with
 * favourable, predominantly sequential access patterns.
 *
 * The reordering is purely organisational (the index-to-dimension mapping is unaffected), so it additionally
 * serves as a canonicalisation: specifications that differ only in the order of equivalent indices, levels or
 * partitions become identical and yield the same rank()/unrank() layout. That is a secondary benefit; the
 * primary purpose is the I/O-friendly traversal order.
 *
 * @tparam Order The storage layout to order for; defaults to column-major.
 * @tparam Partitions A range of partitions, each a range of levels, each level a range of index positions.
 * @tparam Signs A single Symmetry (applied to every partition), a random-access range of per-partition
 *         Symmetry flags, or the all_symmetric_t sentinel.
 * @param partitions [in,out] The partition specification to reorder in place.
 * @param signs [in,out] An optional per-partition sign array, reordered in lockstep with the partitions.
 */
template< IndexOrder Order      = IndexOrder::ColumnMajor, std::ranges::range Partitions,
		  partition_signs Signs = all_symmetric_t >
void sort_partitions(Partitions &&partitions, Signs &&signs = all_symmetric_t{}) {
	details::sort_partitions< Order == IndexOrder::ColumnMajor >(partitions, signs);
}

} // namespace tpack
