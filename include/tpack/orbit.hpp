// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/details/binomial.hpp>
#include <tpack/details/level_columns_view.hpp>

#include <cstddef>
#include <iterator>
#include <numeric>
#include <ranges>
#include <type_traits>
#include <vector>

namespace tpack {


/**
 * @brief Counts the orbits of a symmetric indexing, i.e. the number of distinct canonical representatives.
 *
 * Every partition is treated as an independent fully symmetric group acting on its columns. A column is the
 * tuple of indices obtained by taking one index from each level at the same position, so its combined range
 * (the @e effective @e dimension) is the product of the per-level dimensions. Choosing @c num_cols such
 * columns up to permutation is counting multisets, hence
 * @f$\binom{\text{effective\_dim} + \text{num\_cols} - 1}{\text{num\_cols}}@f$ representatives per partition.
 * The total number of orbits is the product across all partitions.
 *
 * @tparam Dimensions A random-access range mapping each index to its dimension.
 * @tparam Partitions A range of partitions, each a range of levels, each level a range of index positions.
 * @param dims The dimension (extent) of every index of the tensor.
 * @param partitions The symmetry partitions acting on the tensor's indices.
 * @return The number of orbits, i.e. the number of entries required to store the symmetric tensor.
 *
 * @pre Within every level of a partition, all referenced indices share the same dimension.
 * @see rank, unrank
 */
template< std::ranges::random_access_range Dimensions, std::ranges::range Partitions >
constexpr std::size_t num_orbits(Dimensions &&dims, Partitions &&partitions) {
	using std::ranges::begin;
	using std::ranges::end;
	using std::ranges::size;

	std::size_t num = 1;

	for (auto &&part : partitions) {
		// Effective (combined) dimension of the tuple of indices making up the current partition
		const std::size_t effective_dim =
			std::accumulate(begin(part), end(part), std::size_t(1),
							[&dims](auto val, const auto &sub_part) { return val * dims[*begin(sub_part)]; });
		const std::size_t part_size = size(*begin(part));

		// Non-redundant part of the current partition is the number of part_size-combinations
		// of effective_dim elements with repetition.
		num *= details::binomial(effective_dim + part_size - 1, part_size);
	}

	return num;
}

namespace details {

	/**
	 * @brief Computes the change in the number of column inversions produced by one @c next_permutation step.
	 *
	 * An inversion is a pair of columns that is in ascending order (i.e. out of canonical, non-increasing
	 * order). This returns, modulo @f$2^N@f$, the signed delta that
	 * @c std::ranges::next_permutation(columns, std::greater<>{}) applies to that inversion count, computed
	 * directly from the pivot/suffix structure of the permutation without materialising the full count. The
	 * modular result is exactly what is needed to keep a running transposition counter correct.
	 *
	 * @tparam Columns The column-view type exposing @c num_cols() and indexed column comparison.
	 * @param columns The columns about to be advanced; inspected but not modified.
	 * @return The inversion-count delta (wrapping on @c std::size_t) of the next permutation step.
	 */
	template< typename Columns > constexpr std::size_t next_permutation_inversion_delta(Columns &columns) {
		const std::size_t num_cols = columns.num_cols();
		if (num_cols == 0) {
			return 0;
		}

		// next_permutation swaps the element in front of the longest non-decreasing suffix (the pivot) with the
		// largest smaller element in that suffix and then reverses the suffix. Without a pivot, it only reverses.
		std::size_t suffix_begin = num_cols - 1;
		while (suffix_begin > 0 && !(columns[suffix_begin - 1] > columns[suffix_begin])) {
			--suffix_begin;
		}

		std::size_t delta = 0;
		std::size_t run   = 0;
		for (std::size_t i = suffix_begin; i < num_cols; ++i) {
			// The reversal removes all inversions of the suffix, which (as it is sorted) are its pairs of distinct
			// columns
			run = i > suffix_begin && columns[i] == columns[i - 1] ? run + 1 : 0;
			delta -= i - suffix_begin - run;

			// The swap adds one inversion plus one for every suffix element equal to the pivot
			if (suffix_begin > 0 && columns[i] == columns[suffix_begin - 1]) {
				delta += 1;
			}
		}
		if (suffix_begin > 0) {
			delta += 1;
		}

		return delta;
	}

} // namespace details

/**
 * @brief Advances an indexing to the next representative of its symmetry orbit.
 *
 * Starting from any representative, repeated calls enumerate every member of the orbit exactly once and,
 * on the final call, restore @p idx to its canonical representative while returning @c false. Columns of the
 * last partition vary fastest; when a partition exhausts its permutations the enumeration carries over into
 * the next one, so the partitions behave like digits of a mixed-radix counter.
 *
 * @tparam Indexing A random-access range holding one index value per tensor index.
 * @tparam Partitions A range of partitions describing the symmetry.
 * @tparam Counters A range of per-partition counters (defaults to @c std::vector<std::size_t>).
 * @param idx [in,out] The indexing to advance in place.
 * @param parts The symmetry partitions.
 * @param counters Optional pointer to one counter per partition. When supplied and zero-initialised for the
 *        canonical representative, @c counters[i] always equals the number of adjacent column transpositions
 *        of partition @c i separating @p idx from the canonical representative, repeated columns included.
 *        This parity/count information is useful for antisymmetric tensors, where it fixes the sign.
 * @return @c true if @p idx was advanced to a further representative, @c false once the orbit is exhausted
 *         (in which case @p idx has been reset to the canonical representative).
 *
 * @pre If @p counters is non-null it must hold exactly one element per partition.
 * @see is_canonical
 */
template< std::ranges::random_access_range Indexing, std::ranges::range Partitions,
		  std::ranges::range Counters = std::vector< std::size_t > >
constexpr bool next_orbit_representative(Indexing &&idx, Partitions &&parts, Counters *counters = nullptr) {
	using std::ranges::begin;
	using std::ranges::end;
	using std::ranges::rbegin;
	using std::ranges::rend;
	using std::ranges::size;

	assert(!counters || std::ranges::distance(*counters) == std::ranges::distance(parts));

	std::remove_cvref_t< decltype(rbegin(*counters)) > counter_it;
	if (counters) {
		counter_it = rbegin(*counters);
	}

	for (auto &&part_levels : std::ranges::views::reverse(parts)) {
		details::LevelColumnsIndexingView columns(part_levels, idx);

		if (counters) {
			assert(counter_it != rend(*counters));
			*counter_it += details::next_permutation_inversion_delta(columns);
			++counter_it;
		}

		auto [_, has_more] = std::ranges::next_permutation(columns, std::greater<>{});
		if (has_more) {
			return true;
		}
		// wrap around to next partition
	}

	// All candidates have been produced -> indexing is now transformed (back) to canonical representative
	return false;
}

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

} // namespace tpack
