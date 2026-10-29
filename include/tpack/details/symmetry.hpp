// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/details/binomial.hpp>
#include <tpack/symmetry.hpp>

#include <cassert>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>

namespace tpack {

/**
 * @brief The admissible types for the per-partition sign argument: the all_symmetric_t sentinel, a single
 *        Symmetry applied to every partition, or a random-access range of per-partition Symmetry flags.
 *
 * Constraining the sign parameter to this concept lets it coexist unambiguously with the other range
 * parameters of @c rank / @c unrank (effective-index scratch buffers of @c std::size_t), so a sign array and
 * a scratch buffer are never confused during overload resolution.
 */
template< typename T >
concept signs_spec =
	std::same_as<
		std::remove_cvref_t< T >,
		all_symmetric_t > || std::convertible_to< T, Symmetry > || (std::ranges::random_access_range< T > && std::same_as< std::ranges::range_value_t< T >, Symmetry >);

} // namespace tpack

namespace tpack::details {

/**
 * @brief Queries whether partition @p index is antisymmetric.
 *
 * @tparam Signs A type satisfying @c signs_spec.
 * @param signs The sign argument (sentinel, a single Symmetry, or a per-partition range).
 * @param index The partition index to query.
 * @return @c false for the all_symmetric_t sentinel; for a single Symmetry whether it equals
 *         @c Symmetry::Antisymm; otherwise whether @c signs[index] equals @c Symmetry::Antisymm.
 */
template< typename Signs >
constexpr bool antisymmetric_at([[maybe_unused]] const Signs &signs, [[maybe_unused]] std::size_t index) {
	if constexpr (std::is_same_v< std::remove_cvref_t< Signs >, all_symmetric_t >) {
		return false;
	} else if constexpr (!std::ranges::random_access_range< Signs >) {
		// A single (scalar) Symmetry applies to every partition.
		return Symmetry(signs).value == Symmetry::Antisymm;
	} else {
		return signs[index].value == Symmetry::Antisymm;
	}
}

/**
 * @brief Asserts that a range-valued sign argument holds exactly one flag per partition.
 *
 * The sentinel and a single Symmetry apply uniformly to every partition and impose no size constraint; a
 * range must match the partition count. Intended for use inside @c assert only.
 *
 * @tparam Signs A type satisfying @c signs_spec.
 * @tparam Partitions A range of partitions.
 * @param signs The sign argument.
 * @param partitions The partitions the signs apply to.
 */
template< typename Signs, std::ranges::range Partitions >
constexpr void assert_signs_match([[maybe_unused]] const Signs &signs, [[maybe_unused]] const Partitions &partitions) {
	if constexpr (std::ranges::random_access_range< Signs >) {
		assert(static_cast< std::size_t >(std::ranges::distance(signs))
			   == static_cast< std::size_t >(std::ranges::distance(partitions)));
	}
}

/**
 * @brief Number of canonical length-@p k sequences drawn from @p n values.
 *
 * A symmetric partition's canonical columns are non-increasing, so they are size-@p k multisets over @p n
 * values (@f$\binom{n + k - 1}{k}@f$); with no values to draw from (@p n == 0) the only such sequence is the
 * empty one, so the count is 1 for @p k == 0 and 0 otherwise. An antisymmetric partition's columns must be
 * strictly decreasing, so they are plain @p k -combinations without repetition (@f$\binom{n}{k}@f$), which is
 * zero when there are fewer values than columns.
 *
 * This counts both the canonical representatives of a whole partition (@p n the effective column dimension,
 * @p k the number of columns) and, used as a combinatorial-number-system term, the sequences ordered before a
 * given column (@p n that column's effective index, @p k the columns still to be placed).
 *
 * @param n The number of values each column may take.
 * @param k The number of columns in the sequence.
 * @param antisym Whether the partition is antisymmetric.
 * @return The number of canonical length-@p k sequences over @p n values.
 */
constexpr std::size_t canonical_count(std::size_t n, std::size_t k, bool antisym) {
	if (antisym) {
		return n >= k ? binomial(n, k) : 0;
	}

	// With no values to draw from, only the empty sequence is canonical; this also avoids the n + k - 1
	// underflow that would feed binomial a bogus argument.
	if (n == 0) {
		return k == 0 ? 1 : 0;
	}

	return binomial(n + k - 1, k);
}

} // namespace tpack::details
