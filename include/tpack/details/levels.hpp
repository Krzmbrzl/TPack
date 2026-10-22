// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <numeric>
#include <ranges>

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

} // namespace tpack::details
