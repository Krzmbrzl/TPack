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

	/// Computes the change (modulo 2^N) in the number of inversions (pairs of columns in ascending order) caused by
	/// applying std::ranges::next_permutation(columns, std::greater<>{})
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

/// Transforms idx into the next representative of its orbit. Returns false (after transforming idx back into the
/// canonical representative) if there is none.
/// If counters is given, counters[i] is updated such that, if it was zero for the canonical representative, it always
/// equals the number of (adjacent) transpositions of columns of partition i that separate idx from the canonical
/// representative. This also holds for repeated columns.
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
