// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/details/binomial.hpp>

#include <cassert>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>

namespace tpack {

/**
 * @brief Per-partition flag selecting whether a partition is symmetric or antisymmetric.
 *
 * Wraps a two-valued enumerator and converts implicitly from the integer sign of a permutation
 * (@c +1 -> @c Symm, @c -1 -> @c Antisymm), so a sign array can be written either with the enumerators
 * (@c {Symmetry::Antisymm, ...}) or with plain signs (@c {-1, ...}). The underlying values are chosen so that
 * @c static_cast<int> recovers that sign.
 */
struct Symmetry {
	enum Value : int { Symm = 1, Antisymm = -1 };

	Value value;

	constexpr Symmetry(Value v) : value(v) {}

	/// @param sign The integer sign of a permutation; must be exactly @c +1 or @c -1.
	constexpr Symmetry(int sign) : value(sign < 0 ? Antisymm : Symm) { assert(sign == 1 || sign == -1); }

	friend constexpr bool operator==(Symmetry, Symmetry) = default;
};

} // namespace tpack

namespace tpack::details {

/**
 * @brief Sentinel standing in for a per-partition sign array in which every partition is symmetric.
 *
 * The public functions accept an optional sign argument. Its parameter defaults to a value of this type, so
 * callers that do not care about antisymmetry can omit it entirely and still exercise a single, unified
 * implementation.
 */
struct all_symmetric_t {};

inline constexpr all_symmetric_t all_symmetric{};

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
 * @brief Number of canonical representatives of a single partition.
 *
 * A symmetric partition's canonical columns are non-increasing, so representatives are size-@p num_cols
 * multisets over @p effective_dim values (@f$\binom{d + k - 1}{k}@f$). An antisymmetric partition's columns
 * must be strictly decreasing, so representatives are plain @p num_cols -combinations without repetition
 * (@f$\binom{d}{k}@f$), which is zero when there are fewer values than columns.
 *
 * @param effective_dim The effective (combined) dimension of a column.
 * @param num_cols The number of columns in the partition.
 * @param antisym Whether the partition is antisymmetric.
 * @return The number of canonical representatives contributed by the partition.
 */
constexpr std::size_t partition_count(std::size_t effective_dim, std::size_t num_cols, bool antisym) {
	if (antisym) {
		return effective_dim >= num_cols ? binomial(effective_dim, num_cols) : 0;
	}

	return binomial(effective_dim + num_cols - 1, num_cols);
}

/**
 * @brief A single term of the combinatorial-number-system rank of a partition's effective index.
 *
 * For the column currently being ranked, @p remaining_cols is how many columns (including this one) are still
 * to be placed and @p value is this column's effective index. The symmetric ranking addresses non-increasing
 * sequences (@f$\binom{n + k - 1}{k}@f$); the antisymmetric ranking addresses strictly decreasing sequences
 * (@f$\binom{n}{k}@f$, which is zero when @p value < @p remaining_cols).
 *
 * @param value The effective index of the column being ranked.
 * @param remaining_cols The number of columns still to be placed, including the current one.
 * @param antisym Whether the partition is antisymmetric.
 * @return The contribution of this column to the partition rank.
 */
constexpr std::size_t combinatorial_term(std::size_t value, std::size_t remaining_cols, bool antisym) {
	if (antisym) {
		return value >= remaining_cols ? binomial(value, remaining_cols) : 0;
	}

	return binomial(value + remaining_cols - 1, remaining_cols);
}

} // namespace tpack::details
