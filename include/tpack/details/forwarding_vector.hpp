// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <initializer_list>
#include <utility>
#include <vector>

namespace tpack::details {

/**
 * @brief Owns a std::vector and forwards the parts of its interface needed to use it as a random-access range.
 *
 * Serves as the common storage of the partition types. Only the vector's range interface is forwarded, so
 * elements can be accessed and modified but the size is only changed through the derived type's own interface.
 */
template< typename T > class ForwardingVector {
public:
	using value_type      = T;
	using size_type       = std::size_t;
	using iterator        = typename std::vector< T >::iterator;
	using const_iterator  = typename std::vector< T >::const_iterator;
	using reference       = T &;
	using const_reference = const T &;

	ForwardingVector() = default;
	ForwardingVector(std::initializer_list< T > items) : m_items(items) {}
	explicit ForwardingVector(std::vector< T > items) : m_items(std::move(items)) {}

	iterator begin() { return m_items.begin(); }
	iterator end() { return m_items.end(); }
	const_iterator begin() const { return m_items.begin(); }
	const_iterator end() const { return m_items.end(); }
	const_iterator cbegin() const { return m_items.cbegin(); }
	const_iterator cend() const { return m_items.cend(); }

	size_type size() const { return m_items.size(); }
	bool empty() const { return m_items.empty(); }

	reference operator[](size_type pos) { return m_items[pos]; }
	const_reference operator[](size_type pos) const { return m_items[pos]; }

	friend bool operator==(const ForwardingVector &, const ForwardingVector &) = default;

protected:
	std::vector< T > m_items;
};

} // namespace tpack::details
