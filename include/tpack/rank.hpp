// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/details/binomial.hpp>
#include <tpack/orbit.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <ranges>
#include <vector>

namespace tpack {

template< std::ranges::random_access_range Indexing, std::ranges::random_access_range EffectiveIndexing,
		  std::ranges::random_access_range Dimensions, std::ranges::range Partitions >
constexpr std::size_t rank(Indexing &&idx, Dimensions &&dims, Partitions &&parts, EffectiveIndexing &&effective_idx) {
	using std::ranges::begin;
	using std::ranges::end;
	using std::ranges::size;

	std::size_t rank = 0;

	assert(is_canonical(idx, parts));
	assert(size(idx) == size(dims));

	std::size_t stride = 1;
	for (auto &&part_levels : parts) {
		// Convert into effective 1D partition (merge different partition levels)
		const std::size_t effective_size = size(*begin(part_levels));
		assert(effective_idx.size() >= effective_size);
		std::ranges::fill(effective_idx, 0);

		std::size_t effective_dim = 0;
		for (std::size_t k = 0; k < effective_size; ++k) {
			std::size_t col_stride = 1;
			for (auto &&current : part_levels) {
				effective_idx[k] += col_stride * idx[current[k]];
				col_stride *= dims[current[k]];
			}
			// col_stride must be the same value in every iteration
			// -> we just need its value as it describes the total dimension
			// of the entries in our effective partition
			effective_dim = col_stride;
		}

		// If effective_idx is not sorted, idx wasn't canonical
		assert(std::ranges::is_sorted(effective_idx | std::ranges::views::take(effective_size), std::greater<>{}));

		// Compute rank of that effective 1D partition
		std::size_t current_rank = 0;
		for (std::size_t i = 0; i < effective_size; ++i) {
			if (effective_idx[i] == 0) {
				// All subsequent elements must also be zero since we are dealing with
				// a canonical indexing that guarantees to list elements in non-increasing
				// order
				break;
			}

			const std::size_t n = effective_idx[i];
			const std::size_t k = effective_size - i;
			current_rank += details::binomial(n + k - 1, k);
		}

		// Multiply by stride and add to total rank
		rank += stride * current_rank;

		// Update stride by multiplying it by the max rank of the effective 1D partition
		stride *= details::binomial(effective_dim + effective_size - 1, effective_size);
	}

	return rank;
}

template< std::ranges::random_access_range Indexing, std::ranges::random_access_range Dimensions,
		  std::ranges::range Partitions >
std::size_t rank(Indexing &&idx, Dimensions &&dims, Partitions &&parts) {
	using std::ranges::begin;
	using std::ranges::size;

	static thread_local std::vector< std::size_t > effective_idx;
	effective_idx.resize(size(idx));

	return rank(idx, dims, parts, effective_idx);
}



template< std::ranges::random_access_range Indexing, std::ranges::random_access_range EffectiveIndexing,
		  std::ranges::random_access_range Dimensions, std::ranges::range Partitions >
constexpr void unrank(Indexing &&idx, std::size_t rank, Dimensions &&dims, Partitions &&parts,
					  EffectiveIndexing &&effective_idx) {
	using std::ranges::begin;
	using std::ranges::end;
	using std::ranges::size;

	assert(size(idx) == size(dims));

	std::ranges::fill(idx, 0);

	// Compute the stride for the last partition, which is given by the product of
	// the dimensions of all partitions prior to it.
	std::size_t part_stride = 1;
	for (auto &&part_levels : parts | std::ranges::views::take(size(parts) - 1)) {
		std::size_t col_dim        = 1;
		const std::size_t num_cols = size(*begin(part_levels));
		for (auto &&level : part_levels) {
			col_dim *= dims[*begin(level)];
		}

		const std::size_t current_part_dim = details::binomial(col_dim + num_cols - 1, num_cols);
		assert(std::numeric_limits< std::size_t >::max() / current_part_dim >= part_stride);
		part_stride *= current_part_dim;
	}

	bool first = true;
	for (auto &&part_levels : std::ranges::views::reverse(parts)) {
		const std::size_t num_cols = size(*begin(part_levels));
		// The dimension of all columns must be equal so we will
		// simply determine the dimension of the first column
		std::size_t col_dim = 1;
		for (auto &&level : part_levels) {
			col_dim *= dims[*begin(level)];
		}

		if (!first) {
			// Update the partition stride for the current partition by dividing part_stride
			// by the dimension of the current partition
			const std::size_t current_part_dim = details::binomial(col_dim + num_cols - 1, num_cols);
			assert(part_stride % current_part_dim == 0);
			part_stride /= current_part_dim;
			assert(part_stride >= 1);
		} else {
			first = false;
		}

		if (col_dim <= 1) {
			// This partition can only take on a single value (or no value, which would be odd
			// but implies the same behavior). In other words: the associated entries in the
			// indexing have to be zero.
			continue;
		}

		std::size_t current_rank = rank / part_stride;
		rank -= current_rank * part_stride;

		// Unrank current_rank into effective indexing
		[[maybe_unused]] const std::size_t effective_size = size(*begin(part_levels));
		assert(effective_idx.size() >= effective_size);
		std::ranges::fill(effective_idx, 0);

		for (std::size_t col = 0; col < num_cols && current_rank > 0; ++col) {
#ifdef TPACK_UNRANK_BINARY_SEARCH
			const std::size_t min_n = 1;
			const std::size_t max_n = col_dim - 1;

			auto generator = std::ranges::views::reverse(std::ranges::views::iota(min_n, max_n + 1))
							 | std::ranges::views::transform([num_cols, col](std::size_t val) {
								   return details::binomial(val + num_cols - col - 1, num_cols - col);
							   });
			std::size_t num_combinations = 0;
			auto it = std::ranges::partition_point(generator, [&num_combinations, current_rank](std::size_t val) {
				if (val <= current_rank) {
					num_combinations = val;
				}
				return val > current_rank;
			});

			std::size_t n = 0;
			if (it != end(generator)) {
				n = max_n - std::ranges::distance(begin(generator), it);
			}

			assert(num_combinations <= current_rank);
			current_rank -= num_combinations;
			effective_idx[col] = n;
#else
			std::size_t n                     = 0;
			std::size_t num_combinations      = 0;
			std::size_t prev_num_combinations = 0;

			do {
				prev_num_combinations = num_combinations;
				effective_idx[col]    = n;
				++n;
				num_combinations = details::binomial(n + num_cols - col - 1, num_cols - col);
			} while (num_combinations <= current_rank);
			assert(prev_num_combinations <= current_rank);
			current_rank -= prev_num_combinations;
#endif
		}

		assert(current_rank == 0);

		// In order for a 1D indexing consisting of only a single partition to be
		// canonical, it has to be sorted
		assert(std::ranges::is_sorted(effective_idx | std::ranges::views::take(effective_size), std::greater<>{}));


		// Transform effective indexing into actual indexing by splitting
		// effective indices back into their individual elements
		std::vector< std::size_t > level_strides;
		level_strides.emplace_back(1);
		for (auto &&level : part_levels) {
			level_strides.emplace_back(dims[*begin(level)] * level_strides.back());
		}
		level_strides.pop_back();

		std::size_t level = level_strides.size() - 1;
		for (auto &&current : std::ranges::views::reverse(part_levels)) {
			std::size_t col = 0;
			for (auto val : current) {
				idx[val] = effective_idx[col] / level_strides[level];
				effective_idx[col] -= idx[val] * level_strides[level];

				++col;
			}
			--level;
		}
	}

	assert(is_canonical(idx, parts));
}

template< std::ranges::random_access_range Indexing, std::ranges::random_access_range Dimensions,
		  std::ranges::range Partitions >
void unrank(Indexing &&idx, std::size_t rank, Dimensions &&dims, Partitions &&parts) {
	using std::ranges::begin;
	using std::ranges::size;

	static thread_local std::vector< std::size_t > effective_idx;
	effective_idx.resize(size(dims));

	unrank(idx, rank, dims, parts, effective_idx);
}

template< std::ranges::random_access_range Indexing = std::vector< std::size_t >,
		  std::ranges::random_access_range Dimensions, std::ranges::range Partitions >
Indexing unrank(std::size_t rank, Dimensions &&dims, Partitions &&parts) {
	Indexing idx;
	idx.resize(size(dims));

	unrank(idx, rank, dims, parts);

	return idx;
}

} // namespace tpack
