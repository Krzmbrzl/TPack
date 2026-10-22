// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/details/apply_permutation.hpp>
#include <tpack/details/canonical.hpp>
#include <tpack/details/level_columns_view.hpp>
#include <tpack/details/levels.hpp>
#include <tpack/details/symmetry.hpp>

#include <algorithm>
#include <cstddef>
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
 * An antisymmetric partition is canonical only when its columns are @e strictly decreasing: two equal columns
 * make the element vanish, so they are not a canonical representative.
 *
 * @tparam Indexing A random-access range holding one index value per tensor index.
 * @tparam Partitions A range of partitions describing the symmetry.
 * @tparam Signs A single Symmetry (applied to every partition), a random-access range of per-partition
 *         Symmetry flags, or the all_symmetric_t sentinel.
 * @param indexing The indexing to test.
 * @param partitions The symmetry partitions.
 * @param signs A single Symmetry applied to every partition, or one Symmetry per partition; defaults to all
 *        partitions symmetric.
 * @return @c true if @p indexing is canonical, @c false otherwise.
 *
 * @see canonicalize, next_orbit_representative, rank
 */
template< std::ranges::random_access_range Indexing, std::ranges::range Partitions,
		  details::signs_spec Signs = details::all_symmetric_t >
constexpr bool is_canonical(Indexing &&indexing, Partitions &&partitions, const Signs &signs = details::all_symmetric) {
	auto key = [&](std::size_t slot) -> decltype(auto) { return indexing[slot]; };

	return details::is_canonical_by(partitions, signs, [&](const auto &part_levels, std::size_t lhs, std::size_t rhs) {
		return details::reverse_lex_compare(part_levels, lhs, rhs, key);
	});
}

/**
 * @brief Canonicalizes the given indexing and reports the antisymmetric phase.
 *
 * An indexing is canonical when, in every partition, the columns are in non-increasing order. Columns are
 * compared in reverse-lexicographic order: the last (most significant) level decides first, and earlier
 * levels only break ties. Each partition is checked independently; there is no ordering requirement between
 * different partitions.
 *
 * The return value is the phase relating the input to its canonical representative: @c +1 or @c -1 according
 * to the parity of the permutation that sorts the antisymmetric partitions (symmetric partitions never
 * contribute a sign), or @c 0 if the element vanishes because some antisymmetric partition has two equal
 * columns. On a non-vanishing result the indexing is brought into canonical form; when the element vanishes
 * (@c 0) the indexing is left in an unspecified state.
 *
 * @tparam Indexing A random-access range holding one index value per tensor index.
 * @tparam Partitions A range of partitions describing the symmetry.
 * @tparam Signs A single Symmetry (applied to every partition), a random-access range of per-partition
 *         Symmetry flags, or the all_symmetric_t sentinel.
 * @param indexing The indexing to canonicalize in place.
 * @param partitions The symmetry partitions.
 * @param signs A single Symmetry applied to every partition, or one Symmetry per partition; defaults to all
 *        partitions symmetric.
 * @return The phase @c +1, @c -1 or @c 0 (vanishing) relating the input to its canonical representative.
 *
 * @see is_canonical
 */
template< std::ranges::random_access_range Indexing, std::ranges::range Partitions,
		  details::signs_spec Signs = details::all_symmetric_t >
constexpr int canonicalize(Indexing &indexing, Partitions &&partitions, const Signs &signs = details::all_symmetric) {
	using std::ranges::begin;
	using std::ranges::size;

	details::assert_signs_match(signs, partitions);

	int phase              = 1;
	std::size_t part_index = 0;

	for (auto &&part_levels : partitions) {
		const bool antisym         = details::antisymmetric_at(signs, part_index);
		const std::size_t num_cols = details::num_columns(part_levels);
		++part_index;

		auto key = [&](std::size_t slot) -> decltype(auto) { return indexing[slot]; };
		auto cmp = [&](std::size_t lhs, std::size_t rhs) {
			return details::reverse_lex_compare(part_levels, lhs, rhs, key) > 0;
		};

		std::vector< std::size_t > order = details::sorted_index_permutation(num_cols, cmp);

		// Apply the determined order to the columns in place; the swap count gives the permutation's parity.
		details::LevelColumnsIndexingView columns(part_levels, indexing);
		const std::size_t transpositions = details::apply_permutation(order, columns);

		if (antisym) {
			if (transpositions % 2 != 0) {
				phase = -phase;
			}

			auto columns_equal = [&](std::size_t lhs, std::size_t rhs) {
				return details::reverse_lex_compare(part_levels, lhs, rhs, key) == 0;
			};

			// After sorting, equal columns are adjacent; any such pair makes the element vanish.
			auto cols = std::views::iota(std::size_t(0), num_cols);
			if (std::ranges::adjacent_find(cols, columns_equal) != std::ranges::end(cols)) {
				// The indexing is left partially canonicalized; stop processing the remaining partitions.
				phase = 0;
				break;
			}
		}
	}

	return phase;
}

} // namespace tpack
