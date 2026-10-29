// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <numeric>
#include <ranges>
#include <vector>

namespace tpack::details {

/**
 * @brief The number of columns of a partition.
 *
 * Every level of a partition spans the same columns, so the count is the size of its first level. An empty
 * partition (no levels) has no columns.
 *
 * @param part The partition, a range of levels.
 * @return The number of columns, or 0 if @p part has no levels.
 */
template< typename Partition > constexpr std::size_t num_columns(const Partition &part) {
	return std::ranges::empty(part) ? 0 : std::ranges::size(*std::ranges::begin(part));
}

/**
 * @brief The effective (combined) dimension of a partition's columns.
 *
 * Merging the levels of a partition into a single effective index multiplies their per-level dimensions. All
 * slots of one level share a dimension (a precondition of the public API), so each level contributes the
 * dimension of its first slot.
 *
 * @param dims The per-index dimensions.
 * @param part The partition, a range of levels.
 * @return The product of the per-level dimensions (1 for an empty partition).
 */
template< typename Dimensions, typename Partition >
constexpr std::size_t effective_dim(const Dimensions &dims, const Partition &part) {
	return std::accumulate(std::ranges::begin(part), std::ranges::end(part), std::size_t(1),
						   [&dims](std::size_t acc, auto &&level) { return acc * dims[*std::ranges::begin(level)]; });
}

/**
 * @brief The stride of every level within the merged effective index of a partition.
 *
 * The first level is the least significant one (stride 1); each further level is scaled by the dimension of the
 * levels before it.
 *
 * @param dims The per-index dimensions.
 * @param part The partition, a range of levels.
 * @return One stride per level, in level order.
 */
template< typename Dimensions, typename Partition >
constexpr std::vector< std::size_t > level_strides(const Dimensions &dims, const Partition &part) {
	std::vector< std::size_t > strides;
	strides.emplace_back(1);
	for (auto &&level : part) {
		strides.emplace_back(dims[*std::ranges::begin(level)] * strides.back());
	}
	strides.pop_back();

	return strides;
}

/**
 * @brief Splits the effective index of one column back into the indices of its slots.
 *
 * Inverse of merging the levels of a column into a single effective index.
 *
 * @param part The partition, a range of levels.
 * @param strides The level strides of @p part (see level_strides).
 * @param col The column to write.
 * @param effective The effective index of that column.
 * @param indexing [out] Receives the index of every slot in the column.
 */
template< typename Partition, typename Indexing >
constexpr void split_column(const Partition &part, const std::vector< std::size_t > &strides, std::size_t col,
							std::size_t effective, Indexing &indexing) {
	std::size_t level = strides.size();
	for (auto &&current : std::ranges::views::reverse(part)) {
		--level;
		indexing[current[col]] = effective / strides[level];
		effective -= indexing[current[col]] * strides[level];
	}
}

} // namespace tpack::details
