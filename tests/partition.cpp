// SPDX-License-Identifier: BSD-3-Clause

#include "helper.hpp"

#include <tpack/partition.hpp>

#include <cstddef>
#include <tuple>

#include <gtest/gtest.h>


namespace tpack::tests {

struct PartitionTest : testing::TestWithParam< std::tuple< util::PartitionList, util::PartitionList, bool > > {
	using param_tuple = std::tuple< util::PartitionList, util::PartitionList, bool >;
};


TEST_P(PartitionTest, sort_partition) {
	auto [actual, expected, col_major] = GetParam();

	if (col_major) {
		sort_partition_col_major(actual);
	} else {
		sort_partition_row_major(actual);
	}

	EXPECT_EQ(expected, actual);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
	TPack, PartitionTest,
	::testing::Values(
		PartitionTest::param_tuple({ { { 1,  0 } } }, { { { 1, 0 } } }, true),
		PartitionTest::param_tuple({ { { 0,  1 } } }, { { { 1, 0 } } }, true),
		PartitionTest::param_tuple({ { { 0,  1 } } }, { { { 0, 1 } } }, false),
		PartitionTest::param_tuple({ { { 1,  0 } } }, { { { 0, 1 } } }, false),
		PartitionTest::param_tuple({ { { 1,  0 }, { 3, 2 } } }, { { { 1, 0 }, { 3, 2 } } }, true),
		PartitionTest::param_tuple({ { { 0,  1 }, { 2, 3 } } }, { { { 1, 0 }, { 3, 2 } } }, true),
		PartitionTest::param_tuple({ { { 0,  1 }, { 2, 3 } } }, { { { 2, 3 }, { 0, 1 } } }, false),
		PartitionTest::param_tuple({ { { 1,  0 }, { 3, 2 } } }, { { { 2, 3 }, { 0, 1 } } }, false),
		PartitionTest::param_tuple({ { { 1,  0 }, { 3, 2 } }, { { 4 } } }, { { { 1, 0 }, { 3, 2 } }, { { 4 } } }, true),
		PartitionTest::param_tuple({ { { 2,  1 }, { 4, 3 } }, { { 0 } } }, { { { 0 } }, { { 2, 1 }, { 4, 3 } } }, true),
		PartitionTest::param_tuple({ { { 2,  1 }, { 4, 3 } }, { { 0 } } }, { { { 3, 4 }, { 1, 2 } }, { { 0 } } }, false),
		PartitionTest::param_tuple({ { { 1,  0 }, { 3, 2 } }, { { 4 } } }, { { { 4 } }, { { 2, 3 }, { 0, 1 } } }, false)
	)
);
// clang-format on

} // namespace tpack::tests
