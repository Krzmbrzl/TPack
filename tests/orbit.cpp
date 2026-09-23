// SPDX-License-Identifier: BSD-3-Clause

#include "helper.hpp"

#include <tpack/orbit.hpp>

#include <cstddef>
#include <tuple>
#include <vector>

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
		// Scalar
		std::make_tuple(util::make_info({}, {}), 1),
		std::make_tuple(util::make_info_l({ 5 }, { 0 }), 5),
		std::make_tuple(util::make_info({ 5, 5 }, { { { 0 }, { 1 } } }), 25),
		std::make_tuple(util::make_info_l({ 5, 5 }, { 0, 1 }), 15),
		std::make_tuple(util::make_info({ 5, 5, 7 }, { { { 0, 1 } }, { { 2 } } }), 105),
		std::make_tuple(util::make_info_l({ 5, 5, 5 }, { 0, 1, 2 }), 35),
		std::make_tuple(util::make_info_p({ 7, 7, 7, 7 }, { { 0, 1 }, { 2, 3 } }), 1225),
		std::make_tuple(util::make_info({ 5, 5, 5, 8, 8, 10, 10, 3 }, { { { 0, 1, 2 } }, { { 3, 4 }, { 5, 6 } }, { { 7 } } }), 340200),
		// Effective dimension exceeding the range of int
		std::make_tuple(util::make_info_p({ 65536, 65536 }, { { 0 }, { 1 } }), 4294967296)
	)
);
// clang-format on

struct OrbitTest : testing::TestWithParam< std::tuple< std::vector< std::size_t >, util::TensorInfo, bool > > {
	using param_tuple = std::tuple< std::vector< std::size_t >, util::TensorInfo, bool >;
};

TEST_P(OrbitTest, is_canonical) {
	auto [indexing, info, expected] = GetParam();

	bool actual = is_canonical(indexing, info.partitions);

	EXPECT_EQ(expected, actual);
}

TEST_P(OrbitTest, next_orbit_representative) {
	auto [indexing, info, canonical] = GetParam();

	auto orig = indexing;

	std::size_t num_representatives = 0;
	do {
		num_representatives++;
	} while (next_orbit_representative(indexing, info.partitions));

	EXPECT_TRUE(is_canonical(indexing, info.partitions));

	// TODO: have a way to compute the expected number of orbit representatives and compare with that
	EXPECT_GE(num_representatives, 1);
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
		// All pairs have to be checked, not only the first one that differs
		OrbitTest::param_tuple({ 2, 0, 1 }, util::make_info_l({ 3, 3, 3 }, { 0, 1, 2 }), false),
		// If we have multiple levels, we effectively have a partition of l-tuples where l is
		// the number of levels. We still require those l-tuples to be non-ascending. l-tuples
		// are compared in reverse lexicographic order.
		// Tuple elements are assigned column-wise through different levels. That is, for a partition
		// { { 0, 1 }, { 2, 3 } } the index-valued tuples would be (0, 2) and (1, 3). Applied to a
		// specific indexing (5, 4, 6, 7), this would result in the tuples (5, 6) < (4, 7)
		OrbitTest::param_tuple({ 0, 0, 0, 0 }, util::make_info_p({ 3, 3, 3, 3 }, { { 0, 1 }, { 2, 3 } }), true),
		OrbitTest::param_tuple({ 0, 1, 0, 0 }, util::make_info_p({ 3, 3, 3, 3 }, { { 0, 1 }, { 2, 3 } }), false),
		OrbitTest::param_tuple({ 0, 0, 1, 0 }, util::make_info_p({ 3, 3, 3, 3 }, { { 0, 1 }, { 2, 3 } }), true),
		OrbitTest::param_tuple({ 0, 0, 0, 1 }, util::make_info_p({ 3, 3, 3, 3 }, { { 0, 1 }, { 2, 3 } }), false),
		OrbitTest::param_tuple({ 0, 1, 1, 0 }, util::make_info_p({ 3, 3, 3, 3 }, { { 0, 1 }, { 2, 3 } }), true),
		// In case of multiple partitions, they have to be canonical individually without any defined order between them
		OrbitTest::param_tuple({ 1, 0, 0 }, util::make_info({ 3, 3, 3 }, { { { 0, 1 } }, { { 2 } } }), true),
		OrbitTest::param_tuple({ 1, 0, 2 }, util::make_info({ 3, 3, 3 }, { { { 0, 1 } }, { { 2 } } }), true),
		OrbitTest::param_tuple({ 0, 1, 2 }, util::make_info({ 3, 3, 3 }, { { { 0, 1 } }, { { 2 } } }), false),
		OrbitTest::param_tuple({ 1, 0, 0, 1 }, util::make_info({ 3, 3, 3, 3 }, { { { 0, 1 } }, { { 2, 3 } } }), false)
	)
);
// clang-format on

struct OrbitTranspositionTest : testing::TestWithParam< std::tuple< std::vector< std::size_t >, util::TensorInfo > > {
	using param_tuple = std::tuple< std::vector< std::size_t >, util::TensorInfo >;
};

// Number of column pairs of the given partition that are in ascending order, i.e. the number of
// transpositions (of adjacent columns) that separate the current state from the canonical one.
std::size_t num_inversions(const std::vector< std::size_t > &indexing, const util::Partition &partition) {
	auto column = [&](std::size_t col) {
		std::vector< std::size_t > values;
		// Reverse lexicographic comparison -> last level is the most significant one
		for (auto it = partition.rbegin(); it != partition.rend(); ++it) {
			values.push_back(indexing[(*it)[col]]);
		}
		return values;
	};

	std::size_t count = 0;
	for (std::size_t i = 0; i < partition.front().size(); ++i) {
		for (std::size_t j = i + 1; j < partition.front().size(); ++j) {
			if (column(i) < column(j)) {
				++count;
			}
		}
	}

	return count;
}

TEST_P(OrbitTranspositionTest, next_orbit_representative) {
	auto [indexing, info] = GetParam();
	ASSERT_TRUE(is_canonical(indexing, info.partitions));

	std::vector< std::size_t > num_transpositions(info.partitions.size(), 0);

	bool has_more = false;
	do {
		has_more = next_orbit_representative(indexing, info.partitions, &num_transpositions);

		for (std::size_t i = 0; i < info.partitions.size(); ++i) {
			EXPECT_EQ(num_transpositions[i], num_inversions(indexing, info.partitions[i])) << "partition " << i;
		}
	} while (has_more);

	EXPECT_TRUE(is_canonical(indexing, info.partitions));
	for (std::size_t count : num_transpositions) {
		EXPECT_EQ(count, 0);
	}
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
	TPack, OrbitTranspositionTest,
	testing::Values(
		OrbitTranspositionTest::param_tuple({ 2, 1, 0 }, util::make_info_l({ 3, 3, 3 }, { 0, 1, 2 })),
		OrbitTranspositionTest::param_tuple({ 3, 2, 1, 0 }, util::make_info_l({ 4, 4, 4, 4 }, { 0, 1, 2, 3 })),
		// Repeated values
		OrbitTranspositionTest::param_tuple({ 2, 1, 1, 0 }, util::make_info_l({ 3, 3, 3, 3 }, { 0, 1, 2, 3 })),
		OrbitTranspositionTest::param_tuple({ 1, 1, 0, 0 }, util::make_info_l({ 3, 3, 3, 3 }, { 0, 1, 2, 3 })),
		// Multiple levels
		OrbitTranspositionTest::param_tuple({ 1, 0, 2, 2, 2, 0 }, util::make_info_p({ 3, 3, 3, 3, 3, 3 }, { { 0, 1, 2 }, { 3, 4, 5 } })),
		OrbitTranspositionTest::param_tuple({ 0, 1, 1, 2, 1, 1 }, util::make_info_p({ 3, 3, 3, 3, 3, 3 }, { { 0, 1, 2 }, { 3, 4, 5 } })),
		// Multiple partitions
		OrbitTranspositionTest::param_tuple({ 2, 1, 0, 1, 0 }, util::make_info({ 3, 3, 3, 3, 3 }, { { { 0, 1, 2 } }, { { 3, 4 } } }))
	)
);
// clang-format on

} // namespace tpack::tests
