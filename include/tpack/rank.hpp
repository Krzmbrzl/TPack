// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/details/binomial.hpp>
#include <tpack/orbit.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <ranges>
#include <vector>

namespace tpack {

template< std::ranges::random_access_range Indexing, std::ranges::random_access_range Dimensions,
		  std::ranges::range Partitions >
constexpr std::size_t rank(Indexing &&idx, Dimensions &&dims, Partitions &&parts) {
	using std::ranges::begin;
	using std::ranges::end;
	using std::ranges::size;

	std::size_t rank = 0;

	assert(is_canonical(idx, parts));
	assert(size(idx) == size(dims));

	std::vector< std::size_t > effective_idx;

	std::size_t stride = 1;
	for (auto &&part_levels : parts) {
		// Convert into effective 1D partition (merge different partition levels)
		effective_idx.resize(size(*begin(part_levels)));
		std::ranges::fill(effective_idx, 0);

		std::size_t effective_dim = 0;
		for (std::size_t k = 0; k < effective_idx.size(); ++k) {
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
		assert(std::ranges::is_sorted(effective_idx, std::greater<>{}));

		// Compute rank of that effective 1D partition
		std::size_t current_rank = 0;
		for (std::size_t i = 0; i < effective_idx.size(); ++i) {
			if (effective_idx[i] == 0) {
				// All subsequent elements must also be zero since we are dealing with
				// a canonical indexing that guarantees to list elements in non-increasing
				// order
				break;
			}

			const std::size_t n = effective_idx[i];
			const std::size_t k = effective_idx.size() - i;
			current_rank += details::binomial(n + k - 1, k);
		}

		// Multiply by stride and add to total rank
		rank += stride * current_rank;

		// Update stride by multiplying it by the max rank of the effective 1D partition
		stride *= details::binomial(effective_dim + effective_idx.size() - 1, effective_idx.size());
	}

	return rank;
}



template< std::ranges::random_access_range Indexing = std::vector< std::size_t >,
		  std::ranges::random_access_range Dimensions, std::ranges::range Partitions >
constexpr Indexing unrank(std::size_t rank, Dimensions &&dims, Partitions &&parts) {
	using std::ranges::begin;
	using std::ranges::end;
	using std::ranges::size;

	Indexing idx;
	if constexpr (requires(Indexing i) { i.resize(5); }) {
		idx.resize(size(dims));
	}
	assert(size(idx) == size(dims));

	std::ranges::fill(idx, 0);

	// Compute strides
	std::vector< std::size_t > part_strides;
	part_strides.reserve(size(parts));

	std::vector< std::size_t > col_dims;
	col_dims.reserve(size(parts));

	part_strides.emplace_back(1);
	for (auto &&part_levels : parts) {
		// The dimension of all columns must be equal so we will
		// simply determine the dimension of the first column
		col_dims.emplace_back(1);
		const std::size_t num_cols = size(*begin(part_levels));
		for (auto &&level : part_levels) {
			col_dims.back() *= dims[*begin(level)];
		}

		part_strides.emplace_back(part_strides.back() * details::binomial(col_dims.back() + num_cols - 1, num_cols));
	}
	part_strides.pop_back();

	std::size_t part_idx = part_strides.size() - 1;
	for (auto &&part_levels : std::ranges::views::reverse(parts)) {
		if (col_dims[part_idx] <= 1) {
			// This partition can only take on a single value (or no value, which would be odd
			// but implies the same behavior). In other words: the associated entries in the
			// indexing have to be zero.
			--part_idx;
			continue;
		}

		std::size_t current_rank = rank / part_strides[part_idx];
		rank -= current_rank * part_strides[part_idx];

		// Unrank current_rank into effective indexing
		std::vector< std::size_t > effective_idx;
		effective_idx.resize(size(*begin(part_levels)), 0);

		std::size_t num_cols = size(*begin(part_levels));

		for (std::size_t col = 0; col < num_cols && current_rank > 0; ++col) {
#ifdef TPACK_UNRANK_BINARY_SEARCH
			const std::size_t min_n = 1;
			const std::size_t max_n = col_dims[part_idx] - 1;

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
		assert(std::ranges::is_sorted(effective_idx, std::greater<>{}));


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


		--part_idx;
	}

	assert(is_canonical(idx, parts));

	return idx;
}

} // namespace tpack
