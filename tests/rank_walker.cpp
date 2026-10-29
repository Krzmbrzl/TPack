// SPDX-License-Identifier: BSD-3-Clause

#include <tpack/orbit.hpp>
#include <tpack/rank.hpp>
#include <tpack/rank_walker.hpp>
#include <tpack/symmetry.hpp>

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>


namespace tpack::tests {

struct WalkerConfig {
	std::vector< std::size_t > dims;
	std::vector< std::vector< std::vector< std::size_t > > > partitions;
	std::vector< Symmetry > signs;
};

struct RankWalkerTest : testing::TestWithParam< WalkerConfig > {};

TEST_P(RankWalkerTest, matches_unrank) {
	const auto &[dims, partitions, signs] = GetParam();

	const std::size_t count = num_orbits(dims, partitions, signs);
	ASSERT_GT(count, 0);

	RankWalker walker(dims, partitions, signs);
	for (std::size_t r = 0; r < count; ++r) {
		ASSERT_EQ(walker.rank(), r);
		ASSERT_EQ(walker.indexing(), unrank(r, dims, partitions, signs)) << "at rank " << r;

		EXPECT_EQ(walker.next(), r + 1 < count);
	}

	// Wrapped around to the start
	EXPECT_EQ(walker.rank(), 0);
	EXPECT_EQ(walker.indexing(), unrank(0, dims, partitions, signs));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
	TPack, RankWalkerTest,
	::testing::Values(
		// Scalar
		WalkerConfig{ {}, {}, {} },
		WalkerConfig{ { 3 }, { { { 0 } } }, { Symmetry::Symm } },
		WalkerConfig{ { 1 }, { { { 0 } } }, { Symmetry::Symm } },
		WalkerConfig{ { 5, 5 }, { { { 0, 1 } } }, { Symmetry::Symm } },
		WalkerConfig{ { 4, 4, 4 }, { { { 0, 1, 2 } } }, { Symmetry::Symm } },
		// Multiple levels in a partition, with different dimensions
		WalkerConfig{ { 3, 3, 5, 5 }, { { { 0, 1 }, { 2, 3 } } }, { Symmetry::Symm } },
		WalkerConfig{ { 3, 3, 4, 4, 2, 2 }, { { { 0, 1 }, { 2, 3 }, { 4, 5 } } }, { Symmetry::Symm } },
		// Multiple partitions
		WalkerConfig{ { 3, 3, 3, 3 }, { { { 0, 1 } }, { { 2, 3 } } }, { Symmetry::Symm, Symmetry::Symm } },
		WalkerConfig{ { 3, 3, 2, 2, 4 }, { { { 0, 1 } }, { { 2, 3 } }, { { 4 } } }, { Symmetry::Symm, Symmetry::Symm, Symmetry::Symm } },
		// A partition of dimension 1 only has a single value
		WalkerConfig{ { 1, 1, 3, 3 }, { { { 0, 1 } }, { { 2, 3 } } }, { Symmetry::Symm, Symmetry::Symm } },
		// Antisymmetric partitions
		WalkerConfig{ { 4, 4 }, { { { 0, 1 } } }, { Symmetry::Antisymm } },
		WalkerConfig{ { 5, 5, 5 }, { { { 0, 1, 2 } } }, { Symmetry::Antisymm } },
		WalkerConfig{ { 3, 3, 3 }, { { { 0, 1, 2 } } }, { Symmetry::Antisymm } },
		WalkerConfig{ { 2, 2, 3, 3 }, { { { 0, 1 }, { 2, 3 } } }, { Symmetry::Antisymm } },
		WalkerConfig{ { 4, 4, 3, 3 }, { { { 0, 1 } }, { { 2, 3 } } }, { Symmetry::Antisymm, Symmetry::Symm } },
		WalkerConfig{ { 4, 4, 3, 3 }, { { { 0, 1 } }, { { 2, 3 } } }, { Symmetry::Symm, Symmetry::Antisymm } }
	));
// clang-format on

TEST(TPackRankWalker, single_symmetry_for_all_partitions) {
	const std::vector< std::size_t > dims                                     = { 4, 4, 4, 4 };
	const std::vector< std::vector< std::vector< std::size_t > > > partitions = { { { 0, 1 } }, { { 2, 3 } } };

	RankWalker walker(dims, partitions, Symmetry::Antisymm);
	for (std::size_t r = 0; r < num_orbits(dims, partitions, Symmetry::Antisymm); ++r) {
		ASSERT_EQ(walker.indexing(), unrank(r, dims, partitions, Symmetry::Antisymm));
		walker.next();
	}
}

TEST(TPackRankWalker, default_signs_are_symmetric) {
	const std::vector< std::size_t > dims                                     = { 4, 4 };
	const std::vector< std::vector< std::vector< std::size_t > > > partitions = { { { 0, 1 } } };

	RankWalker walker(dims, partitions);
	for (std::size_t r = 0; r < num_orbits(dims, partitions); ++r) {
		ASSERT_EQ(walker.indexing(), unrank(r, dims, partitions));
		walker.next();
	}
}

} // namespace tpack::tests
