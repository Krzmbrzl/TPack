// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <compare>
#include <cstddef>

namespace tpack {

/**
 * @brief A tensor index value carrying a symmetry @p label alongside a local index within that label.
 *
 * Tensors with point-group (abelian) symmetry store each index not as a flat integer but as a pair of a
 * symmetry label (e.g. an irrep) and a local index enumerating the states that share that label. This value
 * type represents that pair and orders it lexicographically -- by @c label first, then by @c local -- which is
 * exactly the ordering the library's canonicalisation uses. Because the compare-only functions (is_canonical,
 * canonicalize, next_orbit_representative, orbit_size) only ever compare and swap index values, they operate on
 * a @c Labeled indexing unchanged.
 *
 * @tparam Label The symmetry label type. Must be three-way comparable (so the defaulted ordering below is
 *         well-formed) and thereby totally ordered.
 */
template< typename Label > struct Labeled {
	Label label;
	std::size_t local;

	friend constexpr auto operator<=>(const Labeled &, const Labeled &) = default;
};

} // namespace tpack
