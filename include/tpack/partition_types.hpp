// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/details/forwarding_vector.hpp>

#include <cstddef>
#include <initializer_list>
#include <utility>

namespace tpack {

/**
 * @brief A level of a partition: a list of tensor index positions.
 *
 * Satisfies the @c level concept.
 */
class Level : public details::ForwardingVector< std::size_t > {
public:
	using ForwardingVector::ForwardingVector;

	/**
	 * @brief The index positions of this level.
	 */
	const std::vector< std::size_t > &indices() const { return m_items; }
};

/**
 * @brief A partition: a list of levels of equal length.
 *
 * Satisfies the @c partition concept.
 */
class Partition : public details::ForwardingVector< Level > {
public:
	using ForwardingVector::ForwardingVector;

	/**
	 * @brief The levels of this partition.
	 */
	const std::vector< Level > &levels() const { return m_items; }

	/**
	 * @brief Appends a level.
	 */
	Partition &add_level(Level level) {
		m_items.push_back(std::move(level));
		return *this;
	}

	std::size_t num_levels() const { return m_items.size(); }

	/**
	 * @brief The number of columns, i.e. the length of every level.
	 */
	std::size_t num_columns() const { return m_items.empty() ? 0 : m_items.front().size(); }
};

/**
 * @brief A list of partitions acting on disjoint tensor indices.
 *
 * Satisfies the @c sortable_partition_list concept.
 */
class PartitionList : public details::ForwardingVector< Partition > {
public:
	using ForwardingVector::ForwardingVector;

	/**
	 * @brief The partitions in this list.
	 */
	const std::vector< Partition > &partitions() const { return m_items; }

	/**
	 * @brief Appends a partition.
	 */
	PartitionList &add_partition(Partition partition) {
		m_items.push_back(std::move(partition));
		return *this;
	}
};

} // namespace tpack
