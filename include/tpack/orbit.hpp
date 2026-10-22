// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/details/binomial.hpp>
#include <tpack/details/level_columns_view.hpp>
#include <tpack/details/symmetry.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <limits>
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
 * An antisymmetric partition instead requires its columns to be strictly decreasing, so its representatives
 * are @c num_cols -combinations @e without repetition, @f$\binom{\text{effective\_dim}}{\text{num\_cols}}@f$
 * (zero when there are fewer values than columns).
 *
 * @tparam Dimensions A random-access range mapping each index to its dimension.
 * @tparam Partitions A range of partitions, each a range of levels, each level a range of index positions.
 * @tparam Signs A single Symmetry (applied to every partition), a random-access range of per-partition
 *         Symmetry flags, or the all_symmetric_t sentinel.
 * @param dims The dimension (extent) of every index of the tensor.
 * @param partitions The symmetry partitions acting on the tensor's indices.
 * @param signs A single Symmetry applied to every partition, or one Symmetry per partition; defaults to all
 *        partitions symmetric.
 * @return The number of orbits, i.e. the number of entries required to store the (anti)symmetric tensor.
 *
 * @pre Within every level of a partition, all referenced indices share the same dimension.
 * @see rank, unrank
 */
template< std::ranges::random_access_range Dimensions, std::ranges::range Partitions,
		  details::signs_spec Signs = details::all_symmetric_t >
constexpr std::size_t num_orbits(Dimensions &&dims, Partitions &&partitions,
								 const Signs &signs = details::all_symmetric) {
	using std::ranges::begin;
	using std::ranges::end;
	using std::ranges::size;

	details::assert_signs_match(signs, partitions);

	std::size_t num        = 1;
	std::size_t part_index = 0;

	for (auto &&part : partitions) {
		// Effective (combined) dimension of the tuple of indices making up the current partition
		const std::size_t effective_dim =
			std::accumulate(begin(part), end(part), std::size_t(1),
							[&dims](auto val, const auto &sub_part) { return val * dims[*begin(sub_part)]; });
		const std::size_t part_size = size(*begin(part));

		num *= details::canonical_count(effective_dim, part_size, details::antisymmetric_at(signs, part_index));
		++part_index;
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
 * When a @p phase pointer is supplied, the running antisymmetric sign is maintained in @c *phase: on each
 * advance it is multiplied by the parity of the column permutation applied to the antisymmetric partitions,
 * so @c *phase always equals @c (-1) raised to the number of column transpositions (over all antisymmetric
 * partitions) separating @p idx from the canonical representative. The caller must initialise @c *phase to
 * that value before the first call; starting the enumeration at the canonical representative, that value is
 * @c +1. Symmetric partitions never contribute a sign.
 *
 * @tparam Indexing A random-access range holding one index value per tensor index.
 * @tparam Partitions A range of partitions describing the symmetry.
 * @tparam Signs A single Symmetry (applied to every partition), a random-access range of per-partition
 *         Symmetry flags, or the all_symmetric_t sentinel.
 * @param idx [in,out] The indexing to advance in place.
 * @param parts The symmetry partitions.
 * @param signs A single Symmetry applied to every partition, or one Symmetry per partition; defaults to all
 *        partitions symmetric.
 * @param phase [in,out] Optional pointer to the running antisymmetric sign (@c +1 or @c -1). When supplied it
 *        must be initialised to the sign of @p idx relative to canonical before the first call (@c +1 when
 *        starting from the canonical representative); it is updated in step with @p idx.
 * @return @c true if @p idx was advanced to a further representative, @c false once the orbit is exhausted
 *         (in which case @p idx has been reset to the canonical representative).
 *
 * @see is_canonical
 */
template< std::ranges::random_access_range Indexing, std::ranges::range Partitions,
		  details::signs_spec Signs = details::all_symmetric_t >
constexpr bool next_orbit_representative(Indexing &&idx, Partitions &&parts,
										 const Signs &signs = details::all_symmetric, int *phase = nullptr) {
	details::assert_signs_match(signs, parts);

	std::size_t part_index = static_cast< std::size_t >(std::ranges::distance(parts));

	for (auto &&part_levels : std::ranges::views::reverse(parts)) {
		--part_index;
		const bool antisym = details::antisymmetric_at(signs, part_index);

		details::LevelColumnsIndexingView columns(part_levels, idx);

		if (phase && antisym && details::next_permutation_inversion_delta(columns) % 2 != 0) {
			*phase = -*phase;
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
 * @brief Compute the number of elements that lie in the same orbit as the provided indexing
 *
 * For an antisymmetric partition any two equal columns make the element vanish (an odd transposition lies in
 * its stabiliser), so such an orbit is reported as size @c 0; otherwise its columns are distinct and the orbit
 * has the full @c num_cols! members.
 *
 * @tparam Indexing A random-access range holding one index value per tensor index.
 * @tparam Partitions A range of partitions describing the symmetry.
 * @tparam Signs A single Symmetry (applied to every partition), a random-access range of per-partition
 *         Symmetry flags, or the all_symmetric_t sentinel.
 * @param idx The reference indexing
 * @param parts The symmetry partitions.
 * @param indexing_canonical Whether @p idx is already guaranteed to be in canonical form.
 * @param signs A single Symmetry applied to every partition, or one Symmetry per partition; defaults to all
 *        partitions symmetric.
 * @return Number of elements in the respective orbit, or @c 0 if the element vanishes by antisymmetry.
 *
 * @pre If @p indexing_canonical is @c true, @p idx must actually be a canonical representative.
 */
template< std::ranges::random_access_range Indexing, std::ranges::range Partitions,
		  details::signs_spec Signs = details::all_symmetric_t >
constexpr std::size_t orbit_size(Indexing &&idx, Partitions &&parts, bool indexing_canonical = false,
								 const Signs &signs = details::all_symmetric) {
	using std::ranges::begin;
	using std::ranges::size;

	details::assert_signs_match(signs, parts);

	std::size_t orb_size   = 1;
	std::size_t part_index = 0;

	for (auto &&part_levels : parts) {
		const bool antisym = details::antisymmetric_at(signs, part_index);
		++part_index;

		details::LevelColumnsIndexingView columns(part_levels, idx);

		const std::size_t num_cols = size(*begin(part_levels));

		// A canonical representation keeps equal columns neighbouring; otherwise sort an index permutation so
		// that equal columns become adjacent (idx itself is left untouched).
		std::vector< std::size_t > order(num_cols);
		std::iota(order.begin(), order.end(), std::size_t(0));
		if (!indexing_canonical) {
			std::ranges::sort(order, [&](std::size_t lhs, std::size_t rhs) { return columns[lhs] < columns[rhs]; });
		}

		// Permuting equal columns yields the same indexing, so the orbit of this fully symmetric partition has
		// num_cols! / prod(repeats_k!) members (a multinomial coefficient). Accumulating it as a product of
		// binomials over the runs of equal columns keeps every intermediate no larger than the final value, so
		// it does not overflow whenever the result itself is representable.
		std::size_t partition_orbit_size = 1;
		std::size_t placed               = 0;
		for (std::size_t i = 0; i < num_cols; ++i) {
			std::size_t repeats = 1;
			while (i + 1 < num_cols && columns[order[i]] == columns[order[i + 1]]) {
				++i;
				++repeats;
			}

			// Equal columns in an antisymmetric partition put a transposition in the stabiliser, so the element
			// vanishes and its orbit holds no distinct representatives.
			if (antisym && repeats >= 2) {
				return 0;
			}

			placed += repeats;
			partition_orbit_size *= details::binomial(placed, repeats);
		}

		assert(std::numeric_limits< decltype(orb_size) >::max() / partition_orbit_size >= orb_size);
		orb_size *= partition_orbit_size;
	}

	return orb_size;
}

} // namespace tpack
