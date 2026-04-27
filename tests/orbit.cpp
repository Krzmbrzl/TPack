// SPDX-License-Identifier: BSD-3-Clause

#include "helper.hpp"

#include <tpack/orbit.hpp>

#include <cstddef>
#include <tuple>

#include <gtest/gtest.h>


namespace tpack::tests {

struct OrbitCountTest : testing::TestWithParam< std::tuple< util::TensorInfo, std::size_t > > {};

TEST_P(OrbitCountTest, num_orbits) {
	auto [info, expected] = GetParam();

	const std::size_t actual = num_orbits(info.dims, info.partitions);

	EXPECT_EQ(expected, actual);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
	TPack, OrbitCountTest,
	testing::Values(
		std::make_tuple(util::make_info_l({ 5 }, { 0 }), 5),
		std::make_tuple(util::make_info({ 5, 5 }, { { { 0 }, { 1 } } }), 25),
		std::make_tuple(util::make_info_l({ 5, 5 }, { 0, 1 }), 15),
		std::make_tuple(util::make_info({ 5, 5, 7 }, { { { 0, 1 } }, { { 2 } } }), 105),
		std::make_tuple(util::make_info_l({ 5, 5, 5 }, { 0, 1, 2 }), 35),
		std::make_tuple(util::make_info_p({ 7, 7, 7, 7 }, { { 0, 1 }, { 2, 3 } }), 1225),
		std::make_tuple(util::make_info({ 5, 5, 5, 8, 8, 10, 10, 3 }, { { { 0, 1, 2 } }, { { 3, 4 }, { 5, 6 } }, { { 7 } } }), 340200)
	)
);
// clang-format on

struct OrbitTest
	: testing::TestWithParam< std::tuple< std::vector< std::size_t >, util::TensorInfo, bool > > {
	using param_tuple = std::tuple< std::vector< std::size_t >, util::TensorInfo, bool >;
};

TEST_P(OrbitTest, is_canonical) {
	auto [indexing, info, expected] = GetParam();

	bool actual = is_canonical(indexing, info.partitions);

	if (expected) {
		EXPECT_TRUE(actual);
	} else {
		EXPECT_FALSE(actual);
	}
}

TEST_P(OrbitTest, next_orbit_representative) {
	auto [indexing, info, is_canonical] = GetParam();

	do {
		std::cout << "Iter" << std::endl;
	} while (next_orbit_representative(indexing, info.partitions));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
	TPack, OrbitTest,
	testing::Values(
		// If there is only a single partition level containing only a single element, indexings are always canonical
		OrbitTest::param_tuple({ 4 }, util::make_info_l({ 8 }, { 0 }), true),
		OrbitTest::param_tuple({ 7 }, util::make_info_l({ 8 }, { 0 }), true),
		// Inside partition levels, we expect the elements to be non-ascending
		OrbitTest::param_tuple({ 0, 0 }, util::make_info_l({ 3, 3 }, { 0, 1 }), true),
		OrbitTest::param_tuple({ 1, 0 }, util::make_info_l({ 3, 3 }, { 0, 1 }), true),
		OrbitTest::param_tuple({ 0, 1 }, util::make_info_l({ 3, 3 }, { 0, 1 }), false),
		OrbitTest::param_tuple({ 1, 0 }, util::make_info_l({ 3, 3 }, { 1, 0 }), false),
		OrbitTest::param_tuple({ 0, 1 }, util::make_info_l({ 3, 3 }, { 1, 0 }), true),
		// If we have multiple levels, we effectively have a partition of l-tuples where l is
		// the number of levels. We still require those l-tuples to be non-ascending. l-tuples
		// are compared in reverse lexicographic order.
		// Tuple elements are assigned column-wise through different levels. That is, for a partition
		// { { 0, 1 }, { 2, 3 } } the index-valued tuples would be (0, 2) and (1, 3). Applied to a
		// specific indexing (5, 4, 6, 7), this would result in the tuples (5, 6) < (4, 7)
		OrbitTest::param_tuple({ 0, 0, 0, 0 }, util::make_info_p({ 3, 3 }, { { 0, 1 }, { 2, 3 } }), true),
		OrbitTest::param_tuple({ 0, 1, 0, 0 }, util::make_info_p({ 3, 3 }, { { 0, 1 }, { 2, 3 } }), false),
		OrbitTest::param_tuple({ 0, 0, 1, 0 }, util::make_info_p({ 3, 3 }, { { 0, 1 }, { 2, 3 } }), true),
		OrbitTest::param_tuple({ 0, 0, 0, 1 }, util::make_info_p({ 3, 3 }, { { 0, 1 }, { 2, 3 } }), false),
		OrbitTest::param_tuple({ 0, 1, 1, 0 }, util::make_info_p({ 3, 3 }, { { 0, 1 }, { 2, 3 } }), true),
		// In case of multiple partitions, they have to be canonical individually without any defined order between them
		OrbitTest::param_tuple({ 1, 0, 0 }, util::make_info({ 3, 3 }, { { { 0, 1 }, { 2 } }, { { 3 } } }), true),
		OrbitTest::param_tuple({ 1, 0, 2 }, util::make_info({ 3, 3 }, { { { 0, 1 }, { 2 } }, { { 3 } } }), true),
		OrbitTest::param_tuple({ 0, 1, 2 }, util::make_info({ 3, 3 }, { { { 0, 1 }, { 2 } }, { { 3 } } }), false)
	)
);
// clang-format on

} // namespace tpack::tests
