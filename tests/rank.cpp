// SPDX-License-Identifier: BSD-3-Clause

#include "helper.hpp"

#include <tpack/orbit.hpp>
#include <tpack/rank.hpp>

#include <array>
#include <cstddef>
#include <set>
#include <span>
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

TEST(TPack, AntisymmetricRankRoundTrip) {
	struct Case {
		util::Dimensions dims;
		PartitionList parts;
		std::vector< Symmetry > signs;
	};

	const std::vector< Case > cases{
		{ { 4, 4 }, { { { 0, 1 } } }, { Symmetry::Antisymm } },
		{ { 5, 5, 5 }, { { { 0, 1, 2 } } }, { Symmetry::Antisymm } },
		{ { 4, 4, 3, 3 }, { { { 0, 1 } }, { { 2, 3 } } }, { Symmetry::Antisymm, Symmetry::Symm } },
		{ { 2, 2, 3, 3 }, { { { 0, 1 }, { 2, 3 } } }, { Symmetry::Antisymm } },
	};

	for (const auto &c : cases) {
		const std::size_t n = num_orbits(c.dims, c.parts, c.signs);

		std::set< std::vector< std::size_t > > seen;
		for (std::size_t r = 0; r < n; ++r) {
			const auto idx = unrank< std::vector< std::size_t > >(r, c.dims, c.parts, c.signs);

			EXPECT_TRUE(is_canonical(idx, c.parts, c.signs));
			EXPECT_EQ(rank(idx, c.dims, c.parts, c.signs), r);
			EXPECT_TRUE(seen.insert(idx).second) << "rank " << r << " not distinct";
		}

		EXPECT_EQ(seen.size(), n);
	}
}

TEST(TPack, AntisymmetricScratchBufferOverloads) {
	// Exercise the explicit scratch-buffer overloads of rank/unrank alongside a sign array. The
	// details::index_scratch constraint must keep the scratch buffer and the sign array unambiguous during
	// overload resolution.
	const util::Dimensions dims{ 5, 5, 5 };
	const PartitionList parts{ { { 0, 1, 2 } } };
	const std::vector< Symmetry > signs{ Symmetry::Antisymm };

	const std::size_t n = num_orbits(dims, parts, signs);
	ASSERT_EQ(n, 10u); // C(5, 3)

	std::vector< std::size_t > effective_idx(dims.size());

	for (std::size_t r = 0; r < n; ++r) {
		std::vector< std::size_t > idx(dims.size());
		unrank(r, idx, dims, parts, effective_idx, signs);

		EXPECT_TRUE(is_canonical(idx, parts, signs));
		EXPECT_EQ(rank(idx, dims, parts, effective_idx, signs), r);
	}
}

TEST(TPack, ExplicitSignSpecForms) {
	// The all_symmetric sentinel, spelled out, must behave exactly like the omitted default, and a single
	// Symmetry like a per-partition range whose every entry is that Symmetry.
	const util::Dimensions dims{ 4, 4, 3, 3 };
	const PartitionList parts{ { { 0, 1 } }, { { 2, 3 } } };

	const std::size_t symmetric = num_orbits(dims, parts);
	EXPECT_EQ(num_orbits(dims, parts, all_symmetric), symmetric);
	EXPECT_EQ(num_orbits(dims, parts, Symmetry::Symm), symmetric);

	const std::vector< Symmetry > all_anti{ Symmetry::Antisymm, Symmetry::Antisymm };
	const std::size_t antisymmetric = num_orbits(dims, parts, Symmetry::Antisymm);
	ASSERT_EQ(antisymmetric, num_orbits(dims, parts, all_anti));

	// rank/unrank round-trip under each spelling of the sign argument.
	for (std::size_t r = 0; r < symmetric; ++r) {
		const auto idx = unrank(r, dims, parts, all_symmetric);
		EXPECT_TRUE(is_canonical(idx, parts, Symmetry::Symm));
		EXPECT_EQ(rank(idx, dims, parts, Symmetry::Symm), r);
	}
	for (std::size_t r = 0; r < antisymmetric; ++r) {
		const auto idx = unrank(r, dims, parts, Symmetry::Antisymm);
		EXPECT_TRUE(is_canonical(idx, parts, all_anti));
		EXPECT_EQ(rank(idx, dims, parts, all_anti), r);
	}
}

TEST(TPack, AlternativeScratchAndIndexingContainers) {
	const util::Dimensions dims{ 5, 5, 5 };
	const PartitionList parts{ { { 0, 1, 2 } } };
	const std::size_t n = num_orbits(dims, parts);

	// A std::span over caller-owned storage satisfies the index_scratch concept, and a std::array works as the
	// indexing at runtime (not just in the constexpr test above).
	std::array< std::size_t, 3 > scratch_storage{};
	const std::span< std::size_t > scratch(scratch_storage);

	for (std::size_t r = 0; r < n; ++r) {
		std::array< std::size_t, 3 > idx{};
		unrank(r, idx, dims, parts, scratch);
		EXPECT_TRUE(is_canonical(idx, parts));
		EXPECT_EQ(rank(idx, dims, parts, scratch), r);
	}

	// An rvalue indexing (a temporary) binds to the forwarding-reference parameter of rank/is_canonical.
	EXPECT_TRUE(is_canonical(std::vector< std::size_t >{ 4, 2, 1 }, parts));
	EXPECT_EQ(rank(std::vector< std::size_t >{ 4, 2, 1 }, dims, parts),
			  rank(std::array< std::size_t, 3 >{ 4, 2, 1 }, dims, parts));
}

namespace {
	// The scratch-buffer overloads are documented as usable in a constant-evaluated context; this exercises
	// them (with std::array scratch, which has no .size() member) at compile time.
	constexpr bool constexpr_roundtrip() {
		std::array< std::size_t, 2 > dims{ 4, 4 };
		std::array< std::array< std::array< std::size_t, 2 >, 1 >, 1 > parts{ { { { { { 0, 1 } } } } } };
		std::array< std::size_t, 2 > scratch{};
		std::array< std::size_t, 2 > idx{};

		for (std::size_t r = 0; r < 10; ++r) { // C(4 + 2 - 1, 2) = 10
			unrank(r, idx, dims, parts, scratch);
			if (rank(idx, dims, parts, scratch) != r) {
				return false;
			}
		}
		return true;
	}

	static_assert(constexpr_roundtrip(), "rank/unrank must round-trip in a constexpr context");
} // namespace

} // namespace tpack::tests
