// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>

namespace tpack::details {

/**
 * @brief Computes the factorial @f$n!@f$.
 *
 * @param n The non-negative integer whose factorial is computed.
 * @return @f$n! = 1 \cdot 2 \cdots n@f$ (and @c 1 for @p n equal to @c 0).
 *
 * @note No overflow checking is performed; @p n must be small enough for the
 *       result to fit into a @c std::size_t.
 */
constexpr std::size_t factorial(std::size_t n) {
	std::size_t result = 1;

	for (std::size_t i = 2; i <= n; ++i) {
		result *= i;
	}

	return result;
}

} // namespace tpack::details
