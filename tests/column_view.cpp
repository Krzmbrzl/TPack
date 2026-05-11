// SPDX-License-Identifier: BSD-3-Clause

#include <tpack/details/level_columns_view.hpp>

#include <algorithm>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

namespace tpack::details::tests {

struct ColViewTest
	: testing::TestWithParam<
		  std::tuple< std::vector< std::vector< std::size_t > >, std::vector< std::vector< std::size_t > > > > {
	using param_tuple =
		std::tuple< std::vector< std::vector< std::size_t > >, std::vector< std::vector< std::size_t > > >;
};

TEST(TPack, ColViewTest_compare) {
	using std::ranges::begin;
	std::vector< std::vector< std::size_t > > levels = { { 0, 1 }, { 1, 0 } };
	LevelColumnsView view(levels);

	ASSERT_GT(view[0], view[1]);
	ASSERT_GE(view[0], view[1]);
	ASSERT_LT(view[1], view[0]);
	ASSERT_LE(view[1], view[0]);
	ASSERT_LE(view[0], view[0]);
	ASSERT_GE(view[0], view[0]);
	ASSERT_EQ(view[0], view[0]);
	ASSERT_NE(view[0], view[1]);
}

TEST_P(ColViewTest, sort) {
	auto [actual, expected] = GetParam();
	LevelColumnsView view(actual);

	std::ranges::sort(view);

	ASSERT_EQ(expected, actual);
}


// clang-format off
INSTANTIATE_TEST_SUITE_P(
	TPack, ColViewTest,
	testing::Values(
		ColViewTest::param_tuple({ { 1, 0 }, { 0, 1 } }, { { 1, 0 }, { 0, 1 } }),
		ColViewTest::param_tuple({ { 0, 1 }, { 1, 0 } }, { { 1, 0 }, { 0, 1 } })
	)
);
// clang-format on

} // namespace tpack::details::tests
