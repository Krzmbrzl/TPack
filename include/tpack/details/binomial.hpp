// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/details/factorial.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <numeric>

namespace tpack::details {

/// Compute the binomial coefficient (n over k)
constexpr std::size_t binomial(std::size_t n, std::size_t k) {
	assert(n >= k);

	// cmp. https://en.wikipedia.org/wiki/Binomial_coefficient#Multiplicative_formula

	// Note: (n over k) == (n over (n - k))
	// so we can choose the lower of the two as the bound
	const std::size_t bound = std::min(k, n - k);

	// We split numerator and denominator so that we can perform the entire
	// computation over integers (individual factors in the multiplicative
	// formula are not guaranteed to be integers).
	// We compute the denominator in one go in advance in order to make the GCD
	// computations as likely to succeed as possible and thus achieve maximal
	// simplification of our numerator to avoid integer overflows as much as possible
	std::size_t numerator   = 1;
	std::size_t denominator = factorial(bound);
	for (std::size_t i = 1; i <= bound; ++i) {
		numerator *= n + 1 - i;

		const std::size_t factor = std::gcd(numerator, denominator);
		numerator /= factor;
		denominator /= factor;
	}

	assert((numerator % denominator) == 0);

	return numerator / denominator;
}

} // namespace tpack::details
