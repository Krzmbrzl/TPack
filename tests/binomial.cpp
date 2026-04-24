// SPDX-License-Identifier: BSD-3-Clause

#include <tpack/details/binomial.hpp>

#include <cstddef>
#include <tuple>

#include <gtest/gtest.h>


namespace tpack::details::tests {

struct BinomialTest : testing::TestWithParam< std::tuple< std::size_t, std::size_t, std::size_t > > {};

TEST_P(BinomialTest, binomial) {
	auto [n, k, expected] = GetParam();

	std::size_t actual = binomial(n, k);

	EXPECT_EQ(expected, actual);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
	TPack, BinomialTest,
	testing::Values(
		std::make_tuple(1, 1, 1),
		std::make_tuple(4, 2, 6),
		std::make_tuple(8, 8, 1),
		std::make_tuple(256, 4, 174792640),
		std::make_tuple(128, 6, 5423611200),
		std::make_tuple(3046, 6, 1103842039076582169),
		std::make_tuple(10000, 5, 832500291625002000),
		std::make_tuple(4099, 6, 6563644220802579456)
	)
);
// clang-format on

} // namespace tpack::details::tests
