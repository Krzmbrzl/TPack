// SPDX-License-Identifier: BSD-3-Clause

#include <tpack/concepts.hpp>
#include <tpack/orbit.hpp>
#include <tpack/partition_types.hpp>
#include <tpack/rank.hpp>

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

namespace tpack::tests {

static_assert(level< Level >);
static_assert(partition< Partition >);
static_assert(partition_list< PartitionList >);
static_assert(sortable_partition_list< PartitionList >);

TEST(TPack, PartitionTypesTest_named_access) {
	const Partition part = { { 0, 1 }, { 2, 3 } };

	EXPECT_EQ(part.num_levels(), 2u);
	EXPECT_EQ(part.num_columns(), 2u);
	EXPECT_EQ(part.levels()[1].indices(), (std::vector< std::size_t >{ 2, 3 }));
	EXPECT_EQ(part[0][1], 1u);

	const PartitionList list = { { { 0, 1 } }, { { 2 } } };
	EXPECT_EQ(list.partitions().size(), 2u);
	EXPECT_EQ(Partition{}.num_columns(), 0u);
}

TEST(TPack, PartitionTypesTest_builder) {
	PartitionList built;
	built.add_partition(Partition{}.add_level({ 0, 1 }).add_level({ 2, 3 }))
		.add_partition(Partition{}.add_level({ 4 }));

	EXPECT_EQ(built, (PartitionList{ { { 0, 1 }, { 2, 3 } }, { { 4 } } }));
}

TEST(TPack, PartitionTypesTest_matches_nested_vectors) {
	const PartitionList list                                              = { { { 0, 1 }, { 2, 3 } }, { { 4 } } };
	const std::vector< std::vector< std::vector< std::size_t > > > nested = { { { 0, 1 }, { 2, 3 } }, { { 4 } } };
	const std::vector< std::size_t > dims                                 = { 3, 3, 2, 2, 4 };

	ASSERT_EQ(num_orbits(dims, list), num_orbits(dims, nested));
	for (std::size_t r = 0; r < num_orbits(dims, list); ++r) {
		EXPECT_EQ(unrank(r, dims, list), unrank(r, dims, nested));
	}
}

} // namespace tpack::tests
