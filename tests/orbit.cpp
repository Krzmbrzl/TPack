// SPDX-License-Identifier: BSD-3-Clause

#include "helper.hpp"

#include <tpack/indexing.hpp>
#include <tpack/orbit.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <set>
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

TEST_P(OrbitTest, next_orbit_representative) {
	auto [indexing, info, canonical] = GetParam();

	std::size_t num_representatives = 0;
	do {
		num_representatives++;
	} while (next_orbit_representative(indexing, info.partitions));

	EXPECT_TRUE(is_canonical(indexing, info.partitions));

	// The loop above restored the canonical representative, so orbit_size() may use its fast path.
	const std::size_t orbit = orbit_size(indexing, info.partitions, all_symmetric, true);
	if (canonical) {
		// Started from the canonical representative, enumeration visits the whole orbit.
		EXPECT_EQ(num_representatives, orbit);
	} else {
		// A non-canonical starting point only enumerates the orbit's remaining suffix before wrapping.
		EXPECT_GE(num_representatives, std::size_t(1));
		EXPECT_LE(num_representatives, orbit);
	}
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
		OrbitTest::param_tuple({ 1, 0, 0, 1 }, util::make_info({ 3, 3, 3, 3 }, { { { 0, 1 } }, { { 2, 3 } } }), false),
		// A scalar (no indices, no partitions) is trivially canonical
		OrbitTest::param_tuple({}, util::make_info({}, {}), true),
		// Reverse lexicographic tie-break: when the most significant level is equal between two
		// columns, the less significant one decides the order
		OrbitTest::param_tuple({ 1, 0, 5, 5 }, util::make_info_p({ 6, 6, 6, 6 }, { { 0, 1 }, { 2, 3 } }), true),
		OrbitTest::param_tuple({ 0, 1, 5, 5 }, util::make_info_p({ 6, 6, 6, 6 }, { { 0, 1 }, { 2, 3 } }), false)
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

	// Treat every partition as antisymmetric so the tracked phase is the parity of the column permutations of
	// all partitions combined, i.e. (-1) raised to the total number of column inversions.
	const std::vector< Symmetry > signs(info.partitions.size(), Symmetry::Antisymm);

	OrbitStep step{ 1, true };
	do {
		step = next_orbit_representative(indexing, info.partitions, signs, step.phase);

		std::size_t total_inversions = 0;
		for (std::size_t i = 0; i < info.partitions.size(); ++i) {
			total_inversions += num_inversions(indexing, info.partitions[i]);
		}
		EXPECT_EQ(step.phase, total_inversions % 2 == 0 ? 1 : -1);
	} while (step.has_more);

	EXPECT_TRUE(is_canonical(indexing, info.partitions));
	EXPECT_EQ(step.phase, 1);
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

struct OrbitEnumerationTest
	: testing::TestWithParam< std::tuple< std::vector< std::size_t >, util::TensorInfo, std::size_t > > {
	using param_tuple = std::tuple< std::vector< std::size_t >, util::TensorInfo, std::size_t >;
};

// Starting from the canonical representative, next_orbit_representative must visit every member of the
// orbit exactly once and then return to the canonical representative.
TEST_P(OrbitEnumerationTest, next_orbit_representative) {
	auto [indexing, info, expected_count] = GetParam();
	ASSERT_TRUE(is_canonical(indexing, info.partitions));

	const auto canonical = indexing;

	std::set< std::vector< std::size_t > > seen;
	std::size_t count = 0;
	OrbitStep step{ 1, true };
	do {
		EXPECT_TRUE(seen.insert(indexing).second) << "representative visited more than once";
		++count;
		step = next_orbit_representative(indexing, info.partitions);
	} while (step.has_more);

	EXPECT_EQ(count, expected_count);
	EXPECT_EQ(indexing, canonical);
	EXPECT_TRUE(is_canonical(indexing, info.partitions));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
	TPack, OrbitEnumerationTest,
	testing::Values(
		// Distinct values -> n! representatives
		OrbitEnumerationTest::param_tuple({ 2, 1, 0 }, util::make_info_l({ 3, 3, 3 }, { 0, 1, 2 }), 6),
		OrbitEnumerationTest::param_tuple({ 3, 2, 1, 0 }, util::make_info_l({ 4, 4, 4, 4 }, { 0, 1, 2, 3 }), 24),
		// Repeated values collapse the orbit (3! / 2! = 3)
		OrbitEnumerationTest::param_tuple({ 1, 1, 0 }, util::make_info_l({ 3, 3, 3 }, { 0, 1, 2 }), 3),
		// All equal -> a single representative
		OrbitEnumerationTest::param_tuple({ 0, 0, 0 }, util::make_info_l({ 3, 3, 3 }, { 0, 1, 2 }), 1),
		// Multiple levels: three distinct column tuples -> 3! representatives
		OrbitEnumerationTest::param_tuple({ 1, 0, 2, 2, 2, 0 }, util::make_info_p({ 3, 3, 3, 3, 3, 3 }, { { 0, 1, 2 }, { 3, 4, 5 } }), 6),
		// Multiple partitions: the orbit sizes multiply (3! * 2! = 12)
		OrbitEnumerationTest::param_tuple({ 2, 1, 0, 1, 0 }, util::make_info({ 3, 3, 3, 3, 3 }, { { { 0, 1, 2 } }, { { 3, 4 } } }), 12),
		// Scalar: a single representative
		OrbitEnumerationTest::param_tuple({}, util::make_info({}, {}), 1)
	)
);
// clang-format on

struct OrbitSizeTest
	: testing::TestWithParam< std::tuple< std::vector< std::size_t >, util::TensorInfo, bool, std::size_t > > {
	using param_tuple = std::tuple< std::vector< std::size_t >, util::TensorInfo, bool, std::size_t >;
};

// orbit_size reports the number of orbit members of a given representative, canonical or not, without
// modifying the indexing. For a canonical representative the result matches the enumeration count checked
// by OrbitEnumerationTest as well as the indexing_canonical fast path.
TEST_P(OrbitSizeTest, orbit_size) {
	auto [indexing, info, is_canon, expected] = GetParam();

	const auto orig = indexing;

	EXPECT_EQ(orbit_size(indexing, info.partitions), expected);
	EXPECT_EQ(indexing, orig) << "orbit_size must not modify the indexing";

	if (is_canon) {
		ASSERT_TRUE(is_canonical(indexing, info.partitions));
		EXPECT_EQ(orbit_size(indexing, info.partitions, all_symmetric, true), expected);
	}
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
	TPack, OrbitSizeTest,
	testing::Values(
		// Canonical representatives with known orbit sizes (cf. OrbitEnumerationTest)
		OrbitSizeTest::param_tuple({ 2, 1, 0 }, util::make_info_l({ 3, 3, 3 }, { 0, 1, 2 }), true, 6),
		OrbitSizeTest::param_tuple({ 3, 2, 1, 0 }, util::make_info_l({ 4, 4, 4, 4 }, { 0, 1, 2, 3 }), true, 24),
		// Repeated values collapse the orbit (3! / 2! = 3)
		OrbitSizeTest::param_tuple({ 1, 1, 0 }, util::make_info_l({ 3, 3, 3 }, { 0, 1, 2 }), true, 3),
		// All equal -> a single representative
		OrbitSizeTest::param_tuple({ 0, 0, 0 }, util::make_info_l({ 3, 3, 3 }, { 0, 1, 2 }), true, 1),
		// Multiple levels: three distinct column tuples -> 3!
		OrbitSizeTest::param_tuple({ 1, 0, 2, 2, 2, 0 }, util::make_info_p({ 3, 3, 3, 3, 3, 3 }, { { 0, 1, 2 }, { 3, 4, 5 } }), true, 6),
		// Multiple partitions: the orbit sizes multiply (3! * 2! = 12)
		OrbitSizeTest::param_tuple({ 2, 1, 0, 1, 0 }, util::make_info({ 3, 3, 3, 3, 3 }, { { { 0, 1, 2 } }, { { 3, 4 } } }), true, 12),
		// A single-column partition always yields an orbit of size one
		OrbitSizeTest::param_tuple({ 4 }, util::make_info_l({ 8 }, { 0 }), true, 1),
		// Scalar: a single member
		OrbitSizeTest::param_tuple({}, util::make_info({}, {}), true, 1),
		// Non-canonical representatives must give the same orbit size (the default path sorts internally)
		OrbitSizeTest::param_tuple({ 0, 1, 2 }, util::make_info_l({ 3, 3, 3 }, { 0, 1, 2 }), false, 6),
		OrbitSizeTest::param_tuple({ 0, 1, 1 }, util::make_info_l({ 3, 3, 3 }, { 0, 1, 2 }), false, 3),
		OrbitSizeTest::param_tuple({ 0, 1, 2, 0, 1 }, util::make_info({ 3, 3, 3, 3, 3 }, { { { 0, 1, 2 } }, { { 3, 4 } } }), false, 12),
		OrbitSizeTest::param_tuple({ 0, 1, 2, 2, 2, 0 }, util::make_info_p({ 3, 3, 3, 3, 3, 3 }, { { 0, 1, 2 }, { 3, 4, 5 } }), false, 6)
	)
);
// clang-format on

// A fully symmetric partition of 21 columns whose values split into runs of 11 and 10 has an orbit of
// 21! / (11! 10!) = C(21, 11) = 352716 members. That result fits into size_t even though the intermediate
// 21! does not, so a naive factorial-based computation would overflow here.
TEST(TPack, OrbitSizeTest_no_intermediate_overflow) {
	constexpr std::size_t num_cols = 21;

	std::vector< std::size_t > level(num_cols);
	std::iota(level.begin(), level.end(), std::size_t(0));
	const util::TensorInfo info = util::make_info_l(std::vector< std::size_t >(num_cols, 2), level);

	// Canonical (non-increasing) representative: eleven 1s followed by ten 0s.
	std::vector< std::size_t > indexing(num_cols, 0);
	std::fill(indexing.begin(), indexing.begin() + 11, std::size_t(1));
	ASSERT_TRUE(is_canonical(indexing, info.partitions));

	EXPECT_EQ(orbit_size(indexing, info.partitions), std::size_t(352716));
	EXPECT_EQ(orbit_size(indexing, info.partitions, all_symmetric, true), std::size_t(352716));
}

TEST(TPack, AntisymmetricOrbitCount) {
	const util::Dimensions dims{ 4, 4 };
	const util::PartitionList parts{ { { 0, 1 } } };

	EXPECT_EQ(num_orbits(dims, parts, std::vector< Symmetry >{ Symmetry::Antisymm }), 6u); // C(4, 2)
	EXPECT_EQ(num_orbits(dims, parts, std::vector< Symmetry >{ Symmetry::Symm }), 10u);    // C(5, 2)

	// Fewer distinct values than columns -> no antisymmetric representatives exist
	EXPECT_EQ(num_orbits(util::Dimensions{ 2, 2, 2 }, util::PartitionList{ { { 0, 1, 2 } } },
						 std::vector< Symmetry >{ Symmetry::Antisymm }),
			  0u);
}

TEST(TPack, SignSpecScalarAndIntForms) {
	const util::Dimensions dims{ 4, 4, 3, 3 };
	const util::PartitionList parts{ { { 0, 1 } }, { { 2, 3 } } };

	// A single Symmetry applies to every partition, matching an all-antisymmetric range.
	EXPECT_EQ(num_orbits(dims, parts, Symmetry::Antisymm),
			  num_orbits(dims, parts, std::vector< Symmetry >{ Symmetry::Antisymm, Symmetry::Antisymm }));

	// The enumerators (and hence a range or scalar) convert implicitly from the sign of a permutation.
	EXPECT_EQ(num_orbits(dims, parts, std::vector< Symmetry >{ -1, 1 }),
			  num_orbits(dims, parts, std::vector< Symmetry >{ Symmetry::Antisymm, Symmetry::Symm }));
	EXPECT_EQ(num_orbits(dims, parts, -1), num_orbits(dims, parts, Symmetry::Antisymm));
}

TEST(TPack, SymmetryStaticCastToInt) {
	EXPECT_EQ(static_cast< int >(Symmetry{ Symmetry::Symm }), 1);
	EXPECT_EQ(static_cast< int >(Symmetry{ Symmetry::Antisymm }), -1);

	// Casting a Symmetry matches casting its underlying value, and round-trips through the integer-sign
	// constructor.
	for (const Symmetry s : { Symmetry::Symm, Symmetry::Antisymm }) {
		EXPECT_EQ(static_cast< int >(s), static_cast< int >(s.value));
		EXPECT_EQ(Symmetry(static_cast< int >(s)), s);
	}
}

TEST(TPack, AntisymmetricOrbitSize) {
	const util::PartitionList parts{ { { 0, 1 } } };
	const std::vector< Symmetry > anti{ Symmetry::Antisymm };

	EXPECT_EQ(orbit_size(std::vector< std::size_t >{ 2, 1 }, parts, anti, false), 2u); // distinct columns -> 2!
	EXPECT_EQ(orbit_size(std::vector< std::size_t >{ 1, 1 }, parts, anti, false), 0u); // equal columns -> vanishes
}

TEST(TPack, NextOrbitPhaseMixedSymmetry) {
	// Partition 0 is symmetric, partition 1 antisymmetric. The tracked phase must reflect only the
	// antisymmetric partition's column inversions; permuting the symmetric partition contributes no sign.
	const util::PartitionList parts{ { { 0, 1 } }, { { 2, 3 } } };
	const std::vector< Symmetry > signs{ Symmetry::Symm, Symmetry::Antisymm };

	std::vector< std::size_t > indexing{ 3, 1, 2, 0 }; // canonical for both partitions
	ASSERT_TRUE(is_canonical(indexing, parts, signs));

	OrbitStep step{ 1, true };
	std::size_t members = 0;
	do {
		step = next_orbit_representative(indexing, parts, signs, step.phase);
		++members;

		// Only the antisymmetric partition's inversions fix the sign.
		const std::size_t antisym_inversions = num_inversions(indexing, parts[1]);
		EXPECT_EQ(step.phase, antisym_inversions % 2 == 0 ? 1 : -1);

		// canonicalize must agree on the phase it reports for the same indexing.
		std::vector< std::size_t > copy = indexing;
		EXPECT_EQ(canonicalize(copy, parts, signs), step.phase);
	} while (step.has_more);

	EXPECT_EQ(members, 4u); // 2! (symmetric) * 2! (antisymmetric)
	EXPECT_TRUE(is_canonical(indexing, parts, signs));
	EXPECT_EQ(step.phase, 1);
}

} // namespace tpack::tests
