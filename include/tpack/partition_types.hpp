// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/details/forwarding_vector.hpp>

#include <cstddef>
#include <initializer_list>
#include <utility>
#include <vector>

namespace tpack {

/**
 * @brief A level of a partition: a list of tensor index positions.
 *
 * Satisfies the @c level concept.
 *
 * @tparam T The numeric type of the index positions.
 */
template< typename T = std::size_t > class Level : public details::ForwardingVector< T > {
public:
	Level() = default;
	Level(std::initializer_list< T > items) : Level::ForwardingVector(items) {}
	explicit Level(std::vector< T > items) : Level::ForwardingVector(std::move(items)) {}

	/**
	 * @brief The index positions of this level.
	 */
	const std::vector< T > &indices() const { return this->m_items; }
};

/**
 * @brief A partition: a list of levels of equal length.
 *
 * Satisfies the @c partition concept.
 *
 * @tparam T The numeric type of the index positions.
 */
template< typename T = std::size_t > class Partition : public details::ForwardingVector< Level< T > > {
public:
	Partition() = default;
	Partition(std::initializer_list< Level< T > > items) : Partition::ForwardingVector(items) {}
	explicit Partition(std::vector< Level< T > > items) : Partition::ForwardingVector(std::move(items)) {}

	/**
	 * @brief The levels of this partition.
	 */
	const std::vector< Level< T > > &levels() const { return this->m_items; }

	/**
	 * @brief Appends a level.
	 */
	Partition &add_level(Level< T > level) {
		this->m_items.push_back(std::move(level));
		return *this;
	}

	std::size_t num_levels() const { return this->m_items.size(); }

	/**
	 * @brief The number of columns, i.e. the length of every level.
	 */
	std::size_t num_columns() const { return this->m_items.empty() ? 0 : this->m_items.front().size(); }
};

/**
 * @brief A list of partitions acting on disjoint tensor indices.
 *
 * Satisfies the @c sortable_partition_list concept.
 *
 * @tparam T The numeric type of the index positions.
 */
template< typename T = std::size_t > class PartitionList : public details::ForwardingVector< Partition< T > > {
public:
	PartitionList() = default;
	PartitionList(std::initializer_list< Partition< T > > items) : PartitionList::ForwardingVector(items) {}
	explicit PartitionList(std::vector< Partition< T > > items) : PartitionList::ForwardingVector(std::move(items)) {}

	/**
	 * @brief The partitions in this list.
	 */
	const std::vector< Partition< T > > &partitions() const { return this->m_items; }

	/**
	 * @brief Appends a partition.
	 */
	PartitionList &add_partition(Partition< T > partition) {
		this->m_items.push_back(std::move(partition));
		return *this;
	}
};

} // namespace tpack
