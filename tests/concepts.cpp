// SPDX-License-Identifier: BSD-3-Clause

#include <tpack/concepts.hpp>
#include <tpack/labeled.hpp>

#include <array>
#include <cstddef>
#include <forward_list>
#include <list>
#include <map>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace {

using Level         = std::vector< std::size_t >;
using Partition     = std::vector< Level >;
using PartitionList = std::vector< Partition >;

using ArrayPartitionList = std::array< std::array< std::array< std::size_t, 2 >, 1 >, 1 >;

// Tuple-like entries whose get() is only found via ADL, resp. as a member
struct AdlEntry {
	int label;
	std::size_t count;
};
template< std::size_t N > decltype(auto) get(const AdlEntry &entry) {
	if constexpr (N == 0) {
		return (entry.label);
	} else {
		return (entry.count);
	}
}

struct MemberEntry {
	int label;
	std::size_t count;

	template< std::size_t N > decltype(auto) get() const {
		if constexpr (N == 0) {
			return (label);
		} else {
			return (count);
		}
	}
};

struct NotTupleLike {
	int label;
	std::size_t count;
};

} // namespace

template<> struct std::tuple_size< AdlEntry > : std::integral_constant< std::size_t, 2 > {};
template<> struct std::tuple_element< 0, AdlEntry > { using type = int; };
template<> struct std::tuple_element< 1, AdlEntry > { using type = std::size_t; };
template<> struct std::tuple_size< MemberEntry > : std::integral_constant< std::size_t, 2 > {};
template<> struct std::tuple_element< 0, MemberEntry > { using type = int; };
template<> struct std::tuple_element< 1, MemberEntry > { using type = std::size_t; };

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

TEST(TPackConcepts, labels) {
	enum class Irrep { A, B };
	struct Unordered {
		bool operator==(const Unordered &) const = default;
	};

	static_assert(tpack::label< int >);
	static_assert(tpack::label< Irrep >);
	static_assert(tpack::label< std::string >);

	static_assert(!tpack::label< int & >);
	static_assert(!tpack::label< Unordered >);

	auto xor_op = [](int lhs, int rhs) { return lhs ^ rhs; };
	static_assert(tpack::label_combiner< decltype(xor_op), int >);
	static_assert(tpack::label_combiner< decltype(xor_op), long >);
	static_assert(!tpack::label_combiner< decltype(xor_op), std::string >);
	static_assert(!tpack::label_combiner< int, int >);
}

TEST(TPackConcepts, labeled_dimensions) {
	using Dims = std::vector< std::map< int, std::size_t > >;

	static_assert(tpack::labeled_dimensions< Dims, int >);
	static_assert(tpack::labeled_dimensions< std::vector< std::vector< std::pair< int, int > > >, int >);
	static_assert(tpack::labeled_dimensions< std::array< std::array< std::tuple< int, std::size_t >, 2 >, 3 >, int >);

	static_assert(tpack::labeled_dimensions< std::vector< std::vector< AdlEntry > >, int >);
	static_assert(tpack::labeled_dimensions< std::vector< std::vector< MemberEntry > >, int >);
	static_assert(!tpack::labeled_dimensions< std::vector< std::vector< NotTupleLike > >, int >);
	static_assert(!tpack::labeled_dimensions< std::vector< std::vector< std::tuple< int, int, int > > >, int >);

	static_assert(!tpack::labeled_dimensions< Dims, std::string >);
	static_assert(!tpack::labeled_dimensions< std::vector< int >, int >);
	static_assert(!tpack::labeled_dimensions< std::vector< std::vector< int > >, int >);
}

TEST(TPackConcepts, labeled_indexing) {
	using Indexing = std::vector< tpack::Labeled< int > >;

	static_assert(tpack::labeled_indexing< Indexing, int >);
	static_assert(tpack::labeled_indexing< const Indexing &, int >);
	static_assert(!tpack::labeled_indexing< Indexing, long >);
	static_assert(!tpack::labeled_indexing< std::vector< std::size_t >, int >);

	static_assert(tpack::writable_labeled_indexing< Indexing &, int >);
	static_assert(!tpack::writable_labeled_indexing< const Indexing &, int >);
}
