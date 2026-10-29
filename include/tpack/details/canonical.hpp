// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/concepts.hpp>
#include <tpack/details/levels.hpp>
#include <tpack/details/symmetry.hpp>

#include <algorithm>
#include <cassert>
#include <compare>
#include <cstddef>
#include <functional>
#include <ranges>

namespace tpack::details {

/**
 * @brief Compares two columns of a partition reverse-lexicographically over the levels.
 *
 * The last (most significant) level decides first; earlier levels only break ties.
 *
 * @param part_levels The levels of the partition, each a random-access range of slots.
 * @param lhs The index of the left column.
 * @param rhs The index of the right column.
 * @param key Maps a slot to the comparable value of that slot.
 * @return A strong_ordering instance describing the columns order
 */
template< typename Levels, typename Key >
constexpr std::strong_ordering reverse_lex_compare(const Levels &part_levels, std::size_t lhs, std::size_t rhs,
												   Key &&key) {
	for (auto &&level : std::ranges::views::reverse(part_levels)) {
		auto &&left  = key(level[lhs]);
		auto &&right = key(level[rhs]);
		if (left < right) {
			return std::strong_ordering::less;
		}
		if (right < left) {
			return std::strong_ordering::greater;
		}
	}

	return std::strong_ordering::equal;
}

/**
 * @brief Whether a merged effective index is a canonical single-partition sequence.
 *
 * A canonical sequence is non-increasing; an antisymmetric one must in addition be strictly decreasing, since
 * equal columns make the element vanish. Intended for use inside @c assert.
 *
 * @param effective The effective (merged) index sequence of one partition.
 * @param antisym Whether the partition is antisymmetric.
 * @return @c true if @p effective is ordered canonically for the given symmetry.
 */
template< typename Range > constexpr bool is_canonical_sequence(Range &&effective, bool antisym) {
	return std::ranges::is_sorted(effective, std::greater<>{})
		   && (!antisym || std::ranges::adjacent_find(effective) == std::ranges::end(effective));
}

/**
 * @brief Tests whether the columns of every partition are non-increasing under a given column ordering.
 *
 * Columns that compare equal are only allowed in symmetric partitions. Each partition is checked
 * independently.
 *
 * @param partitions The symmetry partitions.
 * @param signs A single Symmetry, a per-partition range of Symmetry flags, or the all_symmetric_t sentinel.
 * @param compare_columns A callable @c std::strong_ordering(const Levels &part_levels, std::size_t lhs, std::size_t
 * rhs) that returns a positive / negative / zero value if column @c lhs is greater / smaller / equal to @c rhs.
 * @return @c true if all partitions are ordered canonically.
 */
template< partition_list Partitions, partition_signs Signs, typename CompareColumns >
constexpr bool is_canonical_by(Partitions &&partitions, const Signs &signs, CompareColumns &&compare_columns) {
	assert_signs_match(signs, partitions);

	std::size_t part_index = 0;
	for (auto &&part_levels : partitions) {
		const bool antisym         = antisymmetric_at(signs, part_index);
		const std::size_t num_cols = num_columns(part_levels);
		++part_index;

		for (std::size_t col = 1; col < num_cols; ++col) {
			const std::strong_ordering order = compare_columns(part_levels, col - 1, col);
			if (order < 0 || (order == 0 && antisym)) {
				return false;
			}
		}
	}

	return true;
}

} // namespace tpack::details
