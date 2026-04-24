// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>

namespace tpack::details {

/// Computes the factorial of n
constexpr std::size_t factorial(std::size_t n) {
	std::size_t result = 1;

	for (std::size_t i = 2; i <= n; ++i) {
		result *= i;
	}

	return result;
}

} // namespace tpack::details
