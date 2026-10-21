// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/details/symmetry.hpp>

#include <cassert>
#include <cstddef>
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
 * @return @c +1 / @c -1 / @c 0 if the left column is greater / smaller / equal.
 */
template< typename Levels, typename Key >
constexpr int reverse_lex_compare(const Levels &part_levels, std::size_t lhs, std::size_t rhs, Key &&key) {
	for (auto &&level : std::ranges::views::reverse(part_levels)) {
		auto &&left  = key(level[lhs]);
		auto &&right = key(level[rhs]);
		if (left < right) {
			return -1;
		}
		if (right < left) {
			return 1;
		}
	}

	return 0;
}

/**
 * @brief Tests whether the columns of every partition are non-increasing under a given column ordering.
 *
 * Columns that compare equal are only allowed in symmetric partitions. Each partition is checked
 * independently.
 *
 * @param partitions The symmetry partitions.
 * @param signs A single Symmetry, a per-partition range of Symmetry flags, or the all_symmetric_t sentinel.
 * @param compare_columns A callable @c int(const Levels &part_levels, std::size_t lhs, std::size_t rhs) that
 *        returns a positive / negative / zero value if column @c lhs is greater / smaller / equal to @c rhs.
 * @return @c true if all partitions are ordered canonically.
 */
template< std::ranges::range Partitions, signs_spec Signs, typename CompareColumns >
constexpr bool is_canonical_by(Partitions &&partitions, const Signs &signs, CompareColumns &&compare_columns) {
	assert_signs_match(signs, partitions);

	std::size_t part_index = 0;
	for (auto &&part_levels : partitions) {
		const bool antisym = antisymmetric_at(signs, part_index);
		const std::size_t num_cols =
			std::ranges::empty(part_levels) ? 0 : std::ranges::size(*std::ranges::begin(part_levels));
		++part_index;

		for (std::size_t col = 1; col < num_cols; ++col) {
			const int order = compare_columns(part_levels, col - 1, col);
			if (order < 0 || (order == 0 && antisym)) {
				return false;
			}
		}
	}

	return true;
}

} // namespace tpack::details
