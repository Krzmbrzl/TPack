// SPDX-License-Identifier: BSD-3-Clause

#include "helper.hpp"

#include <tpack/indexing.hpp>
#include <tpack/orbit.hpp>

#include <cstddef>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>


namespace tpack::tests {

struct IndexingTest : testing::TestWithParam< std::tuple< std::vector< std::size_t >, util::PartitionList, bool > > {
	using param_tuple = std::tuple< std::vector< std::size_t >, util::PartitionList, bool >;
};

TEST_P(IndexingTest, is_canonical) {
	auto [indexing, partitions, is_canonical] = GetParam();

	EXPECT_EQ(tpack::is_canonical(indexing, partitions), is_canonical);
}

TEST_P(IndexingTest, canonicalize) {
	auto [indexing, partitions, is_canonical] = GetParam();

	if (!is_canonical) {
		canonicalize(indexing, partitions);
	}

	ASSERT_TRUE(tpack::is_canonical(indexing, partitions));

	auto canonicals_indexing = indexing;

	ASSERT_TRUE(tpack::is_canonical(canonicals_indexing, partitions));

	do {
		auto copy = indexing;

		canonicalize(copy, partitions);

		EXPECT_EQ(copy, canonicals_indexing);
	} while (next_orbit_representative(indexing, partitions));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
	TPack, IndexingTest,
	testing::Values(
		// A scalar (no indices, no partitions) is trivially canonical
		IndexingTest::param_tuple({}, {}, true),
		// A single-column partition level is trivially canonical
		IndexingTest::param_tuple({ 4 }, { { { 0 } } }, true),
		// Single partition, single level
		IndexingTest::param_tuple({ 0, 1 }, { { { { 0, 1 } } } }, false),
		IndexingTest::param_tuple({ 1, 0 }, { { { { 0, 1 } } } }, true),
		IndexingTest::param_tuple({ 1, 0, 2 }, { { { { 2, 1, 0 } } } }, false),
		IndexingTest::param_tuple({ 1, 0, 2 }, { { { { 1, 2, 0 } } } }, false),
		IndexingTest::param_tuple({ 2, 1, 0 }, { { { { 0, 1, 2 } } } }, true),
		// Single partition, multiple levels
		IndexingTest::param_tuple({ 1, 2, 0, 3 }, { { { { 0, 1 }, { 2, 3 } } } }, false),
		IndexingTest::param_tuple({ 2, 1, 0, 3 }, { { { { 0, 1 }, { 2, 3 } } } }, false),
		IndexingTest::param_tuple({ 1, 2, 3, 0 }, { { { { 0, 1 }, { 2, 3 } } } }, true),
		// Single partition, multiple levels: most significant level ties, earlier level breaks it
		IndexingTest::param_tuple({ 0, 1, 5, 5 }, { { { { 0, 1 }, { 2, 3 } } } }, false),
		IndexingTest::param_tuple({ 1, 0, 5, 5 }, { { { { 0, 1 }, { 2, 3 } } } }, true),
		// Equal columns are non-increasing, hence canonical
		IndexingTest::param_tuple({ 2, 2 }, { { { { 0, 1 } } } }, true),
		// Multiple, independent partitions
		IndexingTest::param_tuple({ 1, 0, 1, 0 }, { { { { 0, 1 } } }, { { { 2, 3 } } } }, true),
		IndexingTest::param_tuple({ 0, 1, 0, 2 }, { { { { 0, 1 } } }, { { { 2, 3 } } } }, false)
	)
);
// clang-format on

TEST(TPack, AntisymmetricCanonicalizePhase) {
	const util::PartitionList parts{ { { 0, 1, 2 } } };
	const std::vector< bool > anti{ true };

	// Already canonical (strictly decreasing) -> phase +1
	std::vector< std::size_t > a{ 2, 1, 0 };
	EXPECT_EQ(canonicalize(a, parts, anti), 1);
	EXPECT_EQ(a, (std::vector< std::size_t >{ 2, 1, 0 }));

	// One transposition from canonical -> phase -1
	std::vector< std::size_t > b{ 1, 2, 0 };
	EXPECT_EQ(canonicalize(b, parts, anti), -1);
	EXPECT_EQ(b, (std::vector< std::size_t >{ 2, 1, 0 }));

	// Two transpositions from canonical -> phase +1
	std::vector< std::size_t > c{ 0, 2, 1 };
	EXPECT_EQ(canonicalize(c, parts, anti), 1);
	EXPECT_EQ(c, (std::vector< std::size_t >{ 2, 1, 0 }));

	// Repeated columns -> vanishes (phase 0), but still canonicalized
	std::vector< std::size_t > d{ 1, 2, 1 };
	EXPECT_EQ(canonicalize(d, parts, anti), 0);
	EXPECT_EQ(d, (std::vector< std::size_t >{ 2, 1, 1 }));

	// Symmetric partitions never contribute a sign
	std::vector< std::size_t > e{ 0, 2, 1 };
	EXPECT_EQ(canonicalize(e, parts), 1);
}

TEST(TPack, AntisymmetricIsCanonical) {
	const util::PartitionList parts{ { { 0, 1 } } };
	const std::vector< bool > anti{ true };

	EXPECT_TRUE(is_canonical(std::vector< std::size_t >{ 2, 1 }, parts, anti));
	EXPECT_FALSE(is_canonical(std::vector< std::size_t >{ 1, 2 }, parts, anti));
	// Equal columns are not canonical for an antisymmetric partition, but are for a symmetric one
	EXPECT_FALSE(is_canonical(std::vector< std::size_t >{ 1, 1 }, parts, anti));
	EXPECT_TRUE(is_canonical(std::vector< std::size_t >{ 1, 1 }, parts));
}

} // namespace tpack::tests
