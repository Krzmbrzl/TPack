// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/details/binomial.hpp>

#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>

namespace tpack::details {

/**
 * @brief Sentinel standing in for a per-partition sign array in which every partition is symmetric.
 *
 * The public functions accept an optional sign array (one flag per partition, @c true meaning
 * antisymmetric). Its parameter defaults to a value of this type, so callers that do not care about
 * antisymmetry can omit it entirely and still exercise a single, unified implementation.
 */
struct all_symmetric_t {};

inline constexpr all_symmetric_t all_symmetric{};

/**
 * @brief The admissible types for a per-partition sign array: the all_symmetric_t sentinel, or a
 *        random-access range of @c bool flags.
 *
 * Constraining the sign parameter to this concept lets it coexist unambiguously with the other range
 * parameters of @c rank / @c unrank (effective-index scratch buffers of @c std::size_t), so a sign array and
 * a scratch buffer are never confused during overload resolution.
 */
template< typename T >
concept sign_range =
	std::same_as<
		std::remove_cvref_t< T >,
		all_symmetric_t > || (std::ranges::random_access_range< T > && std::same_as< std::ranges::range_value_t< T >, bool >);

/**
 * @brief Queries whether partition @p index is antisymmetric.
 *
 * @tparam Signs Either all_symmetric_t or a random-access range of bool-convertible per-partition flags.
 * @param signs The sign array (or the all_symmetric_t sentinel).
 * @param index The partition index to query.
 * @return @c false for the all_symmetric_t sentinel, otherwise @c signs[index] contextually converted to bool.
 */
template< typename Signs >
constexpr bool antisymmetric_at([[maybe_unused]] const Signs &signs, [[maybe_unused]] std::size_t index) {
	if constexpr (std::is_same_v< std::remove_cvref_t< Signs >, all_symmetric_t >) {
		return false;
	} else {
		return static_cast< bool >(signs[index]);
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
