// SPDX-License-Identifier: BSD-3-Clause

#include "helper.hpp"

#include <tpack/orbit.hpp>
#include <tpack/partition.hpp>
#include <tpack/rank.hpp>

#include <cstddef>
#include <numeric>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>


namespace tpack::tests {

TEST(TPack, PartitionTest_consistency) {
	const util::PartitionList partitions  = { { { 0, 1 } }, { { 2, 3 }, { 4, 5 } }, { { 6 } } };
	const std::vector< std::size_t > dims = { 3, 3, 2, 2, 4, 4, 5 };
	std::size_t total_dim                 = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>{});

	auto to_flat_idx = [&dims](std::size_t i, std::size_t j, std::size_t k, std::size_t l, std::size_t m, std::size_t n,
							   std::size_t o) -> std::size_t {
		return i + dims[0] * j + std::accumulate(dims.begin(), dims.begin() + 2, 1, std::multiplies<>{}) * k
			   + std::accumulate(dims.begin(), dims.begin() + 3, std::size_t(1), std::multiplies<>{}) * l
			   + std::accumulate(dims.begin(), dims.begin() + 4, std::size_t(1), std::multiplies<>{}) * m
			   + std::accumulate(dims.begin(), dims.begin() + 5, std::size_t(1), std::multiplies<>{}) * n
			   + std::accumulate(dims.begin(), dims.begin() + 6, std::size_t(1), std::multiplies<>{}) * o;
	};

	std::vector< int > data(total_dim, 0);
	std::size_t val_counter = 1;
	for (std::size_t rank = 0; rank < num_orbits(dims, partitions); ++rank) {
		auto indexing = unrank(rank, dims, partitions);
		EXPECT_EQ(indexing.size(), dims.size());

		do {
			int &val = data.at(
				to_flat_idx(indexing[0], indexing[1], indexing[2], indexing[3], indexing[4], indexing[5], indexing[6]));
			ASSERT_EQ(val, 0);
			val = val_counter;
		} while (next_orbit_representative(indexing, partitions));

		val_counter++;
	}

	// Verify that data has the expected structure
	ASSERT_TRUE(std::ranges::all_of(data, [](int val) { return val != 0; }));
	ASSERT_EQ(data.at(to_flat_idx(0, 1, 0, 0, 0, 0, 0)), data.at(to_flat_idx(1, 0, 0, 0, 0, 0, 0)));
	ASSERT_EQ(data.at(to_flat_idx(0, 1, 1, 1, 0, 1, 0)), data.at(to_flat_idx(1, 0, 1, 1, 1, 0, 0)));
	ASSERT_EQ(data.at(to_flat_idx(0, 1, 0, 0, 0, 0, 2)), data.at(to_flat_idx(1, 0, 0, 0, 0, 0, 2)));
	ASSERT_EQ(data.at(to_flat_idx(0, 1, 1, 1, 0, 1, 2)), data.at(to_flat_idx(1, 0, 1, 1, 1, 0, 2)));


	auto col_sorted = partitions;
	sort_partition_col_major(col_sorted);
	auto row_sorted = partitions;
	sort_partition_row_major(row_sorted);

	decltype(data)::value_type ref_sum = 0;
	decltype(data)::value_type col_sum = 0;
	decltype(data)::value_type row_sum = 0;

	for (std::size_t rank = 0; rank < num_orbits(dims, partitions); ++rank) {
		const auto ref_indexing = unrank(rank, std::as_const(dims), std::as_const(partitions));
		const auto col_indexing = unrank(rank, std::as_const(dims), std::as_const(col_sorted));
		const auto row_indexing = unrank(rank, std::as_const(dims), std::as_const(row_sorted));

		std::size_t ref_idx = to_flat_idx(ref_indexing[0], ref_indexing[1], ref_indexing[2], ref_indexing[3],
										  ref_indexing[4], ref_indexing[5], ref_indexing[6]);
		std::size_t col_idx = to_flat_idx(col_indexing[0], col_indexing[1], col_indexing[2], col_indexing[3],
										  col_indexing[4], col_indexing[5], col_indexing[6]);
		std::size_t row_idx = to_flat_idx(row_indexing[0], row_indexing[1], row_indexing[2], row_indexing[3],
										  row_indexing[4], row_indexing[5], row_indexing[6]);

		ASSERT_EQ(data.at(ref_idx), rank + 1);

		ref_sum += data.at(ref_idx);
		col_sum += data.at(col_idx);
		row_sum += data.at(row_idx);
	}

	EXPECT_EQ(ref_sum, col_sum);
	EXPECT_EQ(ref_sum, row_sum);
}

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
