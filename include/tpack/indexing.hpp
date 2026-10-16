// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ranges>

namespace tpack {

/**
 * @brief Tests whether an indexing is the canonical representative of its orbit.
 *
 * An indexing is canonical when, in every partition, the columns are in non-increasing order. Columns are
 * compared in reverse-lexicographic order: the last (most significant) level decides first, and earlier
 * levels only break ties. Each partition is checked independently; there is no ordering requirement between
 * different partitions.
 *
 * @tparam Indexing A random-access range holding one index value per tensor index.
 * @tparam Partitions A range of partitions describing the symmetry.
 * @param indexing The indexing to test.
 * @param partitions The symmetry partitions.
 * @return @c true if @p indexing is canonical, @c false otherwise.
 *
 * @see next_orbit_representative, rank
 */
template< std::ranges::random_access_range Indexing, std::ranges::range Partitions >
constexpr bool is_canonical(Indexing &&indexing, Partitions &&partitions) {
	using std::ranges::begin;
	using std::ranges::size;

	for (auto &&part_levels : partitions) {
		const std::size_t num_cols = size(*begin(part_levels));

		// Columns (compared in reverse lexicographic order) have to be non-increasing
		for (std::size_t col = 1; col < num_cols; ++col) {
			for (auto &&level : std::ranges::views::reverse(part_levels)) {
				if (indexing[level[col - 1]] < indexing[level[col]]) {
					return false;
				}
				if (indexing[level[col - 1]] > indexing[level[col]]) {
					break;
				}
			}
		}
	}

	return true;
}

}
