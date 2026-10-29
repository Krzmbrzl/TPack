// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/concepts.hpp>
#include <tpack/details/apply_permutation.hpp>
#include <tpack/details/canonical.hpp>
#include <tpack/details/level_columns_view.hpp>
#include <tpack/details/levels.hpp>
#include <tpack/details/symmetry.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>
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
template< tensor_indexing Indexing, partition_list Partitions, partition_signs Signs = all_symmetric_t >
constexpr bool is_canonical(Indexing &&indexing, Partitions &&partitions, const Signs &signs = all_symmetric) {
	auto key = [&](std::size_t slot) -> decltype(auto) { return indexing[slot]; };

	return details::is_canonical_by(partitions, signs, [&](const auto &part_levels, std::size_t lhs, std::size_t rhs) {
		return details::reverse_lex_compare(part_levels, lhs, rhs, key);
	});
}

namespace details {

	/**
	 * @brief Shared implementation of canonicalize; see the public overloads for the contract.
	 */
	template< mutable_tensor_indexing Indexing, partition_list Partitions, partition_signs Signs >
	constexpr int canonicalize_impl(Indexing &indexing, Partitions &&partitions, const Signs &signs) {
		assert_signs_match(signs, partitions);

		int phase              = 1;
		std::size_t part_index = 0;

		for (auto &&part_levels : partitions) {
			const bool antisym         = antisymmetric_at(signs, part_index);
			const std::size_t num_cols = num_columns(part_levels);
			++part_index;

			auto key = [&](std::size_t slot) -> decltype(auto) { return indexing[slot]; };
			auto cmp = [&](std::size_t lhs, std::size_t rhs) {
				return reverse_lex_compare(part_levels, lhs, rhs, key) > 0;
			};

			std::vector< std::size_t > order = sorted_index_permutation(num_cols, cmp);

			// Apply the determined order to the columns in place; the swap count gives the permutation's parity.
			LevelColumnsIndexingView columns(part_levels, indexing);
			const std::size_t transpositions = apply_permutation(order, columns);

			if (antisym) {
				if (transpositions % 2 != 0) {
					phase = -phase;
				}

				auto columns_equal = [&](std::size_t lhs, std::size_t rhs) {
					return reverse_lex_compare(part_levels, lhs, rhs, key) == 0;
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

} // namespace details

/**
 * @brief Canonicalizes the given indexing for a fully symmetric specification.
 *
 * An indexing is canonical when, in every partition, the columns are in non-increasing order. Columns are
 * compared in reverse-lexicographic order: the last (most significant) level decides first, and earlier
 * levels only break ties. Each partition is checked independently; there is no ordering requirement between
 * different partitions.
 *
 * With every partition symmetric the result is always @c +1, so this overload does not carry a
 * @c [[nodiscard]]: the indexing is canonicalized in place and the phase may be ignored.
 *
 * @tparam Indexing A random-access range holding one index value per tensor index.
 * @tparam Partitions A range of partitions describing the symmetry.
 * @param indexing The indexing to canonicalize in place.
 * @param partitions The symmetry partitions.
 * @return The phase, always @c +1 for a fully symmetric specification.
 *
 * @see is_canonical
 */
template< mutable_tensor_indexing Indexing, partition_list Partitions >
constexpr int canonicalize(Indexing &&indexing, Partitions &&partitions, all_symmetric_t signs = {}) {
	return details::canonicalize_impl(indexing, partitions, signs);
}

/**
 * @brief Canonicalizes the given indexing and reports the antisymmetric phase.
 *
 * Like the fully symmetric overload, but with per-partition signs. The return value is the phase relating the
 * input to its canonical representative: @c +1 or @c -1 according to the parity of the permutation that sorts
 * the antisymmetric partitions (symmetric partitions never contribute a sign), or @c 0 if the element
 * vanishes because some antisymmetric partition has two equal columns. On a non-vanishing result the indexing
 * is brought into canonical form; when the element vanishes (@c 0) the indexing is left in an unspecified
 * state.
 *
 * Because the result can be @c -1 or @c 0, this overload is @c [[nodiscard]]: discarding it would silently
 * drop a sign flip or a vanishing element.
 *
 * @tparam Indexing A random-access range holding one index value per tensor index.
 * @tparam Partitions A range of partitions describing the symmetry.
 * @tparam Signs A single Symmetry (applied to every partition) or a random-access range of per-partition
 *         Symmetry flags.
 * @param indexing The indexing to canonicalize in place.
 * @param partitions The symmetry partitions.
 * @param signs A single Symmetry applied to every partition, or one Symmetry per partition.
 * @return The phase @c +1, @c -1 or @c 0 (vanishing) relating the input to its canonical representative.
 *
 * @see is_canonical
 */
template< mutable_tensor_indexing Indexing, partition_list Partitions, partition_signs Signs >
requires(!std::same_as< std::remove_cvref_t< Signs >, all_symmetric_t >)
	[[nodiscard]] constexpr int canonicalize(Indexing &&indexing, Partitions &&partitions, const Signs &signs) {
	return details::canonicalize_impl(indexing, partitions, signs);
}

} // namespace tpack
