// SPDX-License-Identifier: BSD-3-Clause

#include "helper.hpp"

#include <tpack/orbit.hpp>
#include <tpack/rank.hpp>

#include <cstddef>
#include <set>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>


namespace tpack::tests {

struct RankTest : testing::TestWithParam< std::tuple< std::vector< std::size_t >, util::TensorInfo, std::size_t > > {
	using param_tuple = std::tuple< std::vector< std::size_t >, util::TensorInfo, std::size_t >;
};

TEST_P(RankTest, rank) {
	auto [indexing, info, expected] = GetParam();

	std::size_t actual = rank(indexing, info.dims, info.partitions);

	EXPECT_EQ(expected, actual);
}

TEST_P(RankTest, unrank) {
	auto [expected, info, rank] = GetParam();

	auto indexing = unrank(rank, info.dims, info.partitions);

	EXPECT_EQ(expected, indexing);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
	TPack, RankTest,
	::testing::Values(
		// Scalar (no indices and hence no partitions)
		RankTest::param_tuple({}, util::make_info({}, {}), 0),
		RankTest::param_tuple({ 0 }, util::make_info_l({ 1 }, { 0 }), 0),
		RankTest::param_tuple({ 0 }, util::make_info_l({ 3 }, { 0 }), 0),
		RankTest::param_tuple({ 2 }, util::make_info_l({ 3 }, { 0 }), 2),
		RankTest::param_tuple({ 0, 0 }, util::make_info_p({ 3, 3 }, { { 0 }, { 1 } }), 0),
		RankTest::param_tuple({ 1, 0 }, util::make_info_p({ 3, 3 }, { { 0 }, { 1 } }), 1),
		RankTest::param_tuple({ 2, 0 }, util::make_info_p({ 3, 3 }, { { 0 }, { 1 } }), 2),
		RankTest::param_tuple({ 0, 1 }, util::make_info_p({ 3, 3 }, { { 0 }, { 1 } }), 3),
		RankTest::param_tuple({ 0, 2 }, util::make_info_p({ 3, 3 }, { { 0 }, { 1 } }), 6),
		RankTest::param_tuple({ 1, 2 }, util::make_info_p({ 3, 3 }, { { 0 }, { 1 } }), 7),
		// Note: The order of indices in the partition specification corresponds to which position in the
		// indexing will be taken as the "highest bit".
		RankTest::param_tuple({ 0, 0 }, util::make_info_l({ 3, 3 }, { 0, 1 }), 0),
		RankTest::param_tuple({ 0, 0 }, util::make_info_l({ 3, 3 }, { 1, 0 }), 0),
		RankTest::param_tuple({ 1, 0 }, util::make_info_l({ 3, 3 }, { 0, 1 }), 1),
		RankTest::param_tuple({ 0, 1 }, util::make_info_l({ 3, 3 }, { 1, 0 }), 1),
		RankTest::param_tuple({ 1, 1 }, util::make_info_l({ 3, 3 }, { 0, 1 }), 2),
		RankTest::param_tuple({ 1, 1 }, util::make_info_l({ 3, 3 }, { 1, 0 }), 2),
		RankTest::param_tuple({ 2, 1 }, util::make_info_l({ 3, 3 }, { 0, 1 }), 4),
		RankTest::param_tuple({ 1, 2 }, util::make_info_l({ 3, 3 }, { 1, 0 }), 4),
		// The re-ordering properties generalizes to partitions with multiple levels
		RankTest::param_tuple({ 0, 0, 0, 0 }, util::make_info_p({ 3, 3, 5, 5 }, { { 0, 1 }, { 2, 3 } }), 0),
		RankTest::param_tuple({ 1, 0, 0, 0 }, util::make_info_p({ 3, 3, 5, 5 }, { { 0, 1 }, { 2, 3 } }), 1),
		RankTest::param_tuple({ 0, 0, 1, 0 }, util::make_info_p({ 3, 3, 5, 5 }, { { 2, 3 }, { 0, 1 } }), 1),
		RankTest::param_tuple({ 0, 0, 0, 1 }, util::make_info_p({ 3, 3, 5, 5 }, { { 3, 2 }, { 0, 1 } }), 1),
		RankTest::param_tuple({ 1, 1, 0, 0 }, util::make_info_p({ 3, 3, 5, 5 }, { { 0, 1 }, { 2, 3 } }), 2),
		RankTest::param_tuple({ 2, 0, 0, 0 }, util::make_info_p({ 3, 3, 5, 5 }, { { 0, 1 }, { 2, 3 } }), 3),
		RankTest::param_tuple({ 0, 0, 1, 0 }, util::make_info_p({ 3, 3, 5, 5 }, { { 0, 1 }, { 2, 3 } }), 6),
		RankTest::param_tuple({ 0, 1, 1, 0 }, util::make_info_p({ 3, 3, 5, 5 }, { { 0, 1 }, { 2, 3 } }), 7),
		RankTest::param_tuple({ 0, 2, 1, 0 }, util::make_info_p({ 3, 3, 5, 5 }, { { 0, 1 }, { 2, 3 } }), 8),
		RankTest::param_tuple({ 0, 0, 1, 1 }, util::make_info_p({ 3, 3, 5, 5 }, { { 0, 1 }, { 2, 3 } }), 9),
		RankTest::param_tuple({ 1, 0, 1, 0 }, util::make_info_p({ 3, 3, 5, 5 }, { { 0, 1 }, { 2, 3 } }), 10),
		RankTest::param_tuple({ 2, 1, 1, 0 }, util::make_info_p({ 3, 3, 5, 5 }, { { 0, 1 }, { 2, 3 } }), 16),
		// Also works for multiple partitions
		RankTest::param_tuple({ 0, 0, 0, 0 }, util::make_info({ 3, 3, 3, 3 }, { { { 0, 1 } }, { { 2, 3 } } }), 0),
		RankTest::param_tuple({ 1, 0, 0, 0 }, util::make_info({ 3, 3, 3, 3 }, { { { 0, 1 } }, { { 2, 3 } } }), 1),
		RankTest::param_tuple({ 0, 1, 0, 0 }, util::make_info({ 3, 3, 3, 3 }, { { { 1, 0 } }, { { 2, 3 } } }), 1),
		RankTest::param_tuple({ 0, 0, 1, 0 }, util::make_info({ 3, 3, 3, 3 }, { { { 2, 3 } }, { { 0, 1 } } }), 1),
		RankTest::param_tuple({ 0, 0, 0, 1 }, util::make_info({ 3, 3, 3, 3 }, { { { 3, 2 } }, { { 0, 1 } } }), 1),
		RankTest::param_tuple({ 0, 0, 1, 0 }, util::make_info({ 3, 5, 2, 2 }, { { { 0 }, { 1 } }, { { 2, 3 } } }), 15),
		RankTest::param_tuple({ 0, 1, 2, 2 }, util::make_info({ 3, 5, 3, 3 }, { { { 0 }, { 1 } }, { { 2, 3 } } }), 78),
		RankTest::param_tuple({ 2, 1, 0 }, util::make_info({ 3, 3, 1 }, { { { 0, 1 } }, { { 2 } } }), 4),
		RankTest::param_tuple({ 2, 1, 0, 1 }, util::make_info({ 3, 3, 1, 3 }, { { { 0, 1 } }, { { 2 } }, { { 3 } } }), 10)
	)
);
// clang-format on

struct RankRoundTripTest : testing::TestWithParam< util::TensorInfo > {};

// Over the whole rank range, unrank must yield distinct canonical representatives (a bijection) and
// rank must invert it.
TEST_P(RankRoundTripTest, bijection) {
	const auto info = GetParam();

	const std::size_t count = num_orbits(info.dims, info.partitions);
	ASSERT_GT(count, 0u);

	std::set< std::vector< std::size_t > > seen;
	for (std::size_t r = 0; r < count; ++r) {
		auto indexing = unrank(r, info.dims, info.partitions);

		EXPECT_EQ(indexing.size(), info.dims.size());
		EXPECT_TRUE(is_canonical(indexing, info.partitions)) << "rank " << r;
		EXPECT_EQ(rank(indexing, info.dims, info.partitions), r);
		EXPECT_TRUE(seen.insert(std::move(indexing)).second) << "duplicate indexing for rank " << r;
	}

	EXPECT_EQ(seen.size(), count);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
	TPack, RankRoundTripTest,
	::testing::Values(
		// Scalar
		util::make_info({}, {}),
		// Single index
		util::make_info_l({ 4 }, { 0 }),
		// Single partition, single level
		util::make_info_l({ 3, 3, 3 }, { 0, 1, 2 }),
		// Single partition, two levels with differing dimensions
		util::make_info_p({ 3, 3, 5, 5 }, { { 0, 1 }, { 2, 3 } }),
		// Multiple partitions
		util::make_info({ 3, 3, 3, 3 }, { { { 0, 1 } }, { { 2, 3 } } }),
		// Dimension-1 index (partition that can only take a single value)
		util::make_info({ 3, 3, 1 }, { { { 0, 1 } }, { { 2 } } })
	)
);
// clang-format on

} // namespace tpack::tests
