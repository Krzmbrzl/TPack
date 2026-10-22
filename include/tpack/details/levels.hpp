// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
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

} // namespace tpack::details
