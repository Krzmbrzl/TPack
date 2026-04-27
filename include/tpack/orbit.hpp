// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/details/binomial.hpp>
#include <tpack/details/level_columns_view.hpp>

#include <ranges>
#include <cstddef>
#include <iterator>

namespace tpack {


template<std::ranges::random_access_range Dimensions, std::ranges::range Partitions>
constexpr std::size_t num_orbits(Dimensions &&dims, Partitions &&partitions) {
	using std::ranges::begin;
	using std::ranges::end;
	using std::ranges::size;

	if (std::ranges::empty(dims)) {
		return 0;
	}

	std::size_t num = 1;

	for (auto &&part : partitions) {
		// Effective (combined) dimension of the tuple of indices making up the current partition
		const std::size_t effective_dim = std::accumulate(begin(part), end(part), 1,
				[&dims](auto val, const auto &sub_part) { return val * dims[*begin(sub_part)]; });
		const std::size_t part_size = size(*begin(part));

		// Non-redundant part of the current partition is the number of part_size-combinations
		// of effective_dim elements with repetition.
		num *= details::binomial(effective_dim + part_size - 1, part_size);
	}

	return num;	
}

template<std::ranges::random_access_range Indexing, std::ranges::range Partitions>
constexpr bool next_orbit_representative(Indexing &&idx, Partitions &&parts) {
	using std::ranges::begin;
	using std::ranges::end;
	using std::ranges::size;

	for (auto &&part_levels : std::ranges::views::reverse(parts)) {
		details::LevelColumnsView columns(part_levels, idx);

		auto [_, wrapped_around] = std::ranges::next_permutation(columns.begin(), columns.end());
		if (!wrapped_around) {
			return true;
		}
	}

	return false;
}

template<std::ranges::random_access_range Indexing, std::ranges::range Partitions>
constexpr bool is_canonical(Indexing &&indexing, Partitions &&partitions) {
	using std::ranges::begin;
	using std::ranges::end;

	for (auto &&part_levels : partitions) {
		for (auto &&level : std::ranges::views::reverse(part_levels)) {
			auto prev_idx_it = begin(level);
			auto curr_idx_it = prev_idx_it;
			std::ranges::advance(curr_idx_it, 1);
			for (; curr_idx_it != end(level); ++curr_idx_it, ++prev_idx_it) {
				if (indexing[*prev_idx_it] < indexing[*curr_idx_it]) {
					return false;
				}
				if (indexing[*prev_idx_it] > indexing[*curr_idx_it]) {
					return true;
				}
			}
		}
	}

	return true;
}

}
