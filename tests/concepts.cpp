// SPDX-License-Identifier: BSD-3-Clause

#include <tpack/concepts.hpp>
#include <tpack/labeled.hpp>

#include <array>
#include <cstddef>
#include <forward_list>
#include <list>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using Level         = std::vector< std::size_t >;
using Partition     = std::vector< Level >;
using PartitionList = std::vector< Partition >;

using ArrayPartitionList = std::array< std::array< std::array< std::size_t, 2 >, 1 >, 1 >;

} // namespace

TEST(TPackConcepts, level) {
	static_assert(tpack::level< Level >);
	static_assert(tpack::level< const Level & >);
	static_assert(tpack::level< std::array< std::size_t, 2 > >);
	static_assert(tpack::level< std::span< const std::size_t > >);
	static_assert(tpack::level< std::vector< int > >);

	static_assert(!tpack::level< std::list< std::size_t > >);
	static_assert(!tpack::level< std::vector< std::string > >);
	static_assert(!tpack::level< std::size_t >);
}

TEST(TPackConcepts, partition) {
	static_assert(tpack::partition< Partition >);
	static_assert(tpack::partition< const Partition & >);
	static_assert(tpack::partition< std::array< std::array< std::size_t, 2 >, 1 > >);
	static_assert(tpack::partition< std::list< Level > >);

	static_assert(!tpack::partition< std::forward_list< Level > >);
	static_assert(!tpack::partition< Level >);
	static_assert(!tpack::partition< std::vector< std::list< std::size_t > > >);
}

TEST(TPackConcepts, partition_list) {
	static_assert(tpack::partition_list< PartitionList >);
	static_assert(tpack::partition_list< const PartitionList & >);
	static_assert(tpack::partition_list< ArrayPartitionList >);

	static_assert(!tpack::partition_list< std::forward_list< Partition > >);
	static_assert(!tpack::partition_list< Partition >);
	static_assert(!tpack::partition_list< std::vector< Level > >);
}

TEST(TPackConcepts, sortable_partition_list) {
	static_assert(tpack::sortable_partition_list< PartitionList & >);
	static_assert(tpack::sortable_partition_list< ArrayPartitionList & >);

	static_assert(!tpack::sortable_partition_list< const PartitionList & >);
	static_assert(!tpack::sortable_partition_list< std::list< Partition > >);
}

TEST(TPackConcepts, dimensions) {
	static_assert(tpack::dimensions< std::vector< std::size_t > >);
	static_assert(tpack::dimensions< const std::array< int, 3 > & >);
	static_assert(tpack::dimensions< std::span< const std::size_t > >);

	static_assert(!tpack::dimensions< std::list< std::size_t > >);
	static_assert(!tpack::dimensions< std::vector< std::string > >);
}

TEST(TPackConcepts, indexing) {
	using Labels = std::vector< tpack::Labeled< int > >;

	static_assert(tpack::tensor_indexing< std::vector< std::size_t > >);
	static_assert(tpack::tensor_indexing< const std::vector< std::size_t > & >);
	static_assert(tpack::tensor_indexing< Labels >);
	static_assert(tpack::mutable_tensor_indexing< Labels & >);
	static_assert(!tpack::mutable_tensor_indexing< const Labels & >);
	static_assert(!tpack::tensor_indexing< std::list< std::size_t > >);

	static_assert(tpack::integral_indexing< std::array< std::size_t, 3 > >);
	static_assert(tpack::integral_indexing< const std::vector< int > & >);
	static_assert(!tpack::integral_indexing< Labels >);

	static_assert(tpack::writable_integral_indexing< std::vector< std::size_t > & >);
	static_assert(tpack::writable_integral_indexing< std::span< std::size_t > >);
	static_assert(!tpack::writable_integral_indexing< const std::vector< std::size_t > & >);
	static_assert(!tpack::writable_integral_indexing< Labels >);
}

TEST(TPackConcepts, index_scratch) {
	static_assert(tpack::index_scratch< std::vector< std::size_t > & >);
	static_assert(tpack::index_scratch< std::array< std::size_t, 3 > & >);
	static_assert(tpack::index_scratch< const std::span< std::size_t > & >);

	static_assert(!tpack::index_scratch< std::vector< int > & >);
	static_assert(!tpack::index_scratch< const std::vector< std::size_t > & >);
	static_assert(!tpack::index_scratch< std::list< std::size_t > >);
}

TEST(TPackConcepts, partition_signs) {
	static_assert(tpack::partition_signs< tpack::all_symmetric_t >);
	static_assert(tpack::partition_signs< const tpack::all_symmetric_t & >);
	static_assert(tpack::partition_signs< tpack::Symmetry >);
	static_assert(tpack::partition_signs< tpack::Symmetry::Value >);
	static_assert(tpack::partition_signs< std::vector< tpack::Symmetry > >);
	static_assert(tpack::partition_signs< std::array< tpack::Symmetry, 2 > >);

	static_assert(!tpack::partition_signs< std::vector< int > >);
	static_assert(!tpack::partition_signs< std::vector< std::size_t > >);
	static_assert(!tpack::partition_signs< std::list< tpack::Symmetry > >);
}
