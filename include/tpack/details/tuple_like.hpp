// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tpack::details {

namespace get_lookup {
	// Makes std::get visible next to ADL, the same way structured bindings look up a non-member get
	using std::get;

	template< std::size_t N, typename T > concept has_member_get = requires(T &&t) {
		std::forward< T >(t).template get< N >();
	};

	template< std::size_t N, typename T > concept has_adl_get = requires(T &&t) { get< N >(std::forward< T >(t)); };

	/// Fetches the N-th element of a tuple-like @p t, preferring a member get() as structured bindings do.
	template< std::size_t N, typename T >
		requires has_member_get< N, T > || has_adl_get< N, T > constexpr decltype(auto) get_element(T &&t) {
		if constexpr (has_member_get< N, T >) {
			return std::forward< T >(t).template get< N >();
		} else {
			return get< N >(std::forward< T >(t));
		}
	}
} // namespace get_lookup

using get_lookup::get_element;

/**
 * @brief Whether @p T is tuple-like with exactly @p N elements, i.e. can be decomposed via a structured binding
 *        that uses get_element().
 */
template< typename T, std::size_t N > concept tuple_like_of_size = requires {
	typename std::tuple_size< std::remove_cvref_t< T > >::type;
	requires std::tuple_size< std::remove_cvref_t< T > >::value == N;
};

} // namespace tpack::details
