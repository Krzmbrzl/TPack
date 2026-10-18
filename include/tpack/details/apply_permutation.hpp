// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <iterator>
#include <ranges>
#include <utility>

namespace tpack::details {

/**
 * @brief Applies a permutation to one or more ranges in lockstep.
 *
 * Reorders each of @p ranges so that element @c i afterwards holds the element that was at @c order[i], using
 * the in-place cycle-walk from https://stackoverflow.com/a/838789. Every range is permuted with the same
 * sequence of element swaps, so parallel ranges stay aligned. @p order is consumed: it is left sorted (the
 * identity) on return.
 *
 * @tparam Order A random-access range of indices into the ranges; must hold one entry per element.
 * @tparam Ranges Random-access ranges to permute, each at least as long as @p order.
 * @param order [in,out] The permutation to apply; the entry at position @c i names the element moved there.
 * @param ranges [in,out] The ranges to reorder in lockstep.
 * @return The number of element swaps performed, whose parity is the parity of the permutation.
 */
template< std::ranges::random_access_range Order, std::ranges::random_access_range... Ranges >
constexpr std::size_t apply_permutation(Order &order, Ranges &&... ranges) {
	using std::ranges::begin;
	using std::ranges::size;

	const std::size_t num = static_cast< std::size_t >(size(order));

	auto swap_at = [](auto &&range, std::size_t lhs, std::size_t rhs) {
		using Diff = std::ranges::range_difference_t< decltype(range) >;
		std::ranges::iter_swap(begin(range) + static_cast< Diff >(lhs), begin(range) + static_cast< Diff >(rhs));
	};

	std::size_t transpositions = 0;
	for (std::size_t i = 0; i + 1 < num; ++i) {
		if (order[i] == i) {
			continue;
		}

		std::size_t k = i + 1;
		while (order[k] != i) {
			++k;
		}

		(swap_at(ranges, i, order[i]), ...);
		std::swap(order[i], order[k]);
		++transpositions;
	}

	return transpositions;
}

} // namespace tpack::details
