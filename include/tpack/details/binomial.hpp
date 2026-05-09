// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/details/factorial.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>

namespace tpack::details {

/// Compute the binomial coefficient (n over k)
constexpr std::size_t binomial(std::size_t n, std::size_t k) {
	assert(n >= k);

	if (n == k) {
		return 1;
	}

	if (k == 1) {
		return n;
	}

	// cmp. https://en.wikipedia.org/wiki/Binomial_coefficient#Multiplicative_formula

	// Note: (n over k) == (n over (n - k))
	// so we can choose the lower of the two as the bound
	const std::size_t bound = std::min(k, n - k);

	std::size_t result = n;
	std::size_t denom  = 2;
	for (std::size_t i = 1; i < bound; ++i) {
		// We must not overflow result (that would spoil our computation)
		assert(std::numeric_limits< std::size_t >::max() / (n - i) >= result);

		result *= n - i;

		// In order to avoid integer overflows, we try to simplify the result as soon as possible.
		// We start with the lowest factor of the denominator (2) as this is the first factor we can
		// guarantee that result is divisible by. In fact, we can guarantee that among the set of
		// factors (n), (n - 1), (n - 2), ..., (n - i), one of them must be divisible by i + 1.
		// This is also why we can guarantee that in the bound iterations, we will be able to cleanly
		// divide all bound factors (1, 2, 3, ..., bound).
		while (denom <= bound && result % denom == 0) {
			result /= denom;
			++denom;
		}
	}

	// +1 because we always increment denom after we have divided by it
	assert(denom == bound + 1);

	return result;
}

} // namespace tpack::details
