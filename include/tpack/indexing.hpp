// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <ranges>
#include <vector>

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
 * @see canonicalize, next_orbit_representative, rank
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

/**
 * @brief Canonicalizes the given indexing
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
 *
 * @see is_canonical
 */
template< std::ranges::random_access_range Indexing, std::ranges::range Partitions >
constexpr void canonicalize(Indexing &indexing, Partitions &&partitions) {
	using std::ranges::begin;
	using std::ranges::size;

	for (auto &&part_levels : partitions) {
		const std::size_t num_cols = size(*begin(part_levels));

		auto cmp = [&](std::size_t lhs, std::size_t rhs) {
			for (auto &&level : std::ranges::views::reverse(part_levels)) {
				if (indexing[level[lhs]] != indexing[level[rhs]]) {
					return indexing[level[lhs]] > indexing[level[rhs]];
				}
			}

			return false;
		};

		std::vector< std::size_t > order(num_cols, 0);
		std::iota(order.begin(), order.end(), std::size_t(0));

		std::ranges::sort(order, cmp);

		// Apply determined order to indexing (in-place)
		// cmp.: https://stackoverflow.com/a/838789
		for (std::size_t i = 0; i < num_cols - 1; ++i) {
			if (i == order[i]) {
				continue;
			}

			std::size_t k = 0;
			for (k = i + 1; k < num_cols; ++k) {
				if (order[k] == i) {
					break;
				}
			}

			// Swap whole columns
			for (auto &&level : part_levels) {
				std::swap(indexing[level[i]], indexing[level[order[i]]]);
			}

			std::swap(order[i], order[k]);
		}
	}
}

} // namespace tpack
