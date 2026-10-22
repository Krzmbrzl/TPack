// SPDX-License-Identifier: BSD-3-Clause

#include "helper.hpp"

#include <tpack/indexing.hpp>
#include <tpack/labeled.hpp>
#include <tpack/orbit.hpp>

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>


namespace tpack::tests {

using Lidx = std::vector< Labeled< int > >;

// The Labeled value type only ever gets compared and swapped by is_canonical / canonicalize /
// next_orbit_representative / orbit_size, so these exercise those functions on a labelled indexing unchanged.

TEST(TPack, LabeledOrdering) {
	// Lexicographic: label first, then local.
	EXPECT_TRUE((Labeled< int >{ 0, 5 } < Labeled< int >{ 1, 0 }));
	EXPECT_TRUE((Labeled< int >{ 1, 1 } < Labeled< int >{ 1, 2 }));
	EXPECT_TRUE((Labeled< int >{ 1, 2 } == Labeled< int >{ 1, 2 }));
	EXPECT_TRUE((Labeled< int >{ 2, 0 } > Labeled< int >{ 1, 9 }));
}

TEST(TPack, LabeledIsCanonicalSymmetric) {
	const util::PartitionList parts{ { { 0, 1 } } };

	// Columns must be non-increasing (reverse-lexicographically) for a symmetric partition.
	EXPECT_TRUE(is_canonical(Lidx{ { 1, 0 }, { 0, 5 } }, parts));
	EXPECT_FALSE(is_canonical(Lidx{ { 0, 5 }, { 1, 0 } }, parts));
	// Equal labels, distinct locals: ordered by local.
	EXPECT_TRUE(is_canonical(Lidx{ { 1, 3 }, { 1, 1 } }, parts));
	EXPECT_FALSE(is_canonical(Lidx{ { 1, 1 }, { 1, 3 } }, parts));
	// Equal (label, local) columns are non-increasing, hence canonical for a symmetric partition.
	EXPECT_TRUE(is_canonical(Lidx{ { 1, 2 }, { 1, 2 } }, parts));
}

TEST(TPack, LabeledCanonicalizeAndEnumerateSymmetric) {
	const util::PartitionList parts{ { { 0, 1 } } };

	// Canonicalising any orbit member yields the same (non-increasing) representative, and enumerating the
	// orbit visits exactly its orbit_size members.
	Lidx idx{ { 0, 5 }, { 1, 0 } };
	EXPECT_EQ(canonicalize(idx, parts), 1); // symmetric: phase always +1
	EXPECT_EQ(idx, (Lidx{ { 1, 0 }, { 0, 5 } }));

	const Lidx canonical = idx;
	EXPECT_EQ(orbit_size(canonical, parts, all_symmetric, true), 2u);

	std::size_t visited = 0;
	do {
		Lidx copy = idx;
		canonicalize(copy, parts);
		EXPECT_EQ(copy, canonical);
		++visited;
	} while (next_orbit_representative(idx, parts));
	EXPECT_EQ(visited, 2u);

	// Equal columns form a single-member orbit.
	EXPECT_EQ(orbit_size(Lidx{ { 1, 2 }, { 1, 2 } }, parts, all_symmetric, true), 1u);
}

TEST(TPack, LabeledAntisymmetric) {
	const util::PartitionList parts{ { { 0, 1 } } };
	const std::vector< Symmetry > anti{ Symmetry::Antisymm };

	// Equal labels with distinct locals are allowed and do NOT vanish; the canonical form is strictly
	// decreasing and a swap flips the phase.
	EXPECT_TRUE(is_canonical(Lidx{ { 1, 3 }, { 1, 1 } }, parts, anti));
	EXPECT_FALSE(is_canonical(Lidx{ { 1, 1 }, { 1, 3 } }, parts, anti));
	EXPECT_EQ(orbit_size(Lidx{ { 1, 3 }, { 1, 1 } }, parts, anti, true), 2u);

	Lidx swapped{ { 1, 1 }, { 1, 3 } };
	EXPECT_EQ(canonicalize(swapped, parts, anti), -1);
	EXPECT_EQ(swapped, (Lidx{ { 1, 3 }, { 1, 1 } }));

	// Distinct labels: ordered by label, swap flips the phase.
	Lidx distinct{ { 1, 9 }, { 2, 0 } };
	EXPECT_EQ(canonicalize(distinct, parts, anti), -1);
	EXPECT_EQ(distinct, (Lidx{ { 2, 0 }, { 1, 9 } }));

	// Equal (label, local) columns vanish.
	EXPECT_FALSE(is_canonical(Lidx{ { 1, 2 }, { 1, 2 } }, parts, anti));
	EXPECT_EQ(orbit_size(Lidx{ { 1, 2 }, { 1, 2 } }, parts, anti, false), 0u);
	Lidx vanishing{ { 1, 2 }, { 1, 2 } };
	EXPECT_EQ(canonicalize(vanishing, parts, anti), 0);
}

// A non-integer, totally-ordered label type exercises the genericity of Labeled.
enum class Irrep { A, B, C, D };

TEST(TPack, LabeledGenericLabelType) {
	const util::PartitionList parts{ { { 0, 1 } } };
	using LidxE = std::vector< Labeled< Irrep > >;

	EXPECT_TRUE(is_canonical(LidxE{ { Irrep::C, 0 }, { Irrep::A, 7 } }, parts));
	EXPECT_FALSE(is_canonical(LidxE{ { Irrep::A, 7 }, { Irrep::C, 0 } }, parts));

	LidxE idx{ { Irrep::A, 7 }, { Irrep::C, 0 } };
	EXPECT_EQ(canonicalize(idx, parts), 1);
	EXPECT_EQ(idx, (LidxE{ { Irrep::C, 0 }, { Irrep::A, 7 } }));
}

} // namespace tpack::tests
