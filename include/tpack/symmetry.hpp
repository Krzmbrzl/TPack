// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace tpack {

/**
 * @brief Per-partition flag selecting whether a partition is symmetric or antisymmetric.
 *
 * Wraps a two-valued enumerator whose underlying values are the integer sign of a permutation
 * (@c Symm == @c +1, @c Antisymm == @c -1), so a sign array is written with the enumerators
 * (@c {Symmetry::Antisymm, ...}) and @c static_cast<int> recovers the sign, whether applied to a @c Symmetry
 * or to its @c value.
 */
struct Symmetry {
	enum Value : int { Symm = 1, Antisymm = -1 };

	Value value;

	constexpr Symmetry(Value v) : value(v) {}

	/// @return The integer sign of the symmetry (@c +1 for @c Symm, @c -1 for @c Antisymm).
	constexpr explicit operator int() const { return value; }

	friend constexpr bool operator==(Symmetry, Symmetry) = default;
};

/**
 * @brief Sentinel standing in for a per-partition sign array in which every partition is symmetric.
 *
 * The public functions accept an optional sign argument. Its parameter defaults to a value of this type, so
 * callers that do not care about antisymmetry can omit it entirely and still exercise a single, unified
 * implementation.
 */
struct all_symmetric_t {};

inline constexpr all_symmetric_t all_symmetric{};

} // namespace tpack
