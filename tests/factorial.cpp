// SPDX-License-Identifier: BSD-3-Clause

#include <tpack/details/factorial.hpp>

#include <cstddef>
#include <tuple>

#include <gtest/gtest.h>


namespace tpack::details::tests {

struct FactorialTest : testing::TestWithParam< std::tuple< std::size_t, std::size_t > > {};

TEST_P(FactorialTest, factorial) {
	auto [n, expected] = GetParam();

	std::size_t actual = factorial(n);

	EXPECT_EQ(expected, actual);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
	TPack, FactorialTest,
	testing::Values(
		// The loop starts at 2, so the empty product must still yield 1
		std::make_tuple(0, 1),
		std::make_tuple(1, 1),
		std::make_tuple(2, 2),
		std::make_tuple(5, 120),
		std::make_tuple(10, 3628800),
		std::make_tuple(20, 2432902008176640000)
	)
);
// clang-format on

} // namespace tpack::details::tests
