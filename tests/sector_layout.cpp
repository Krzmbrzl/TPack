// SPDX-License-Identifier: BSD-3-Clause

#include "helper.hpp"

#include <tpack/indexing.hpp>
#include <tpack/labeled.hpp>
#include <tpack/sector_layout.hpp>

#include <cstddef>
#include <map>
#include <set>
#include <vector>

#include <gtest/gtest.h>


namespace tpack::tests {

namespace {

	using Lidx     = std::vector< Labeled< int > >;
	using SlotDims = std::vector< std::map< int, std::size_t > >;

	// Enumerate, by brute force, every non-vanishing labelled indexing that satisfies the selection rule and is
	// canonical according to @p canonical. A strict (antisymmetric) predicate already excludes vanishing tuples.
	template< typename Combine, typename Canonical >
	std::set< Lidx > brute_force(const SlotDims &dims, Combine combine, int identity, int target, Canonical canonical) {
		const std::size_t num_slots = dims.size();

		std::vector< Lidx > options(num_slots);
		for (std::size_t slot = 0; slot < num_slots; ++slot) {
			for (const auto &[label, count] : dims[slot]) {
				for (std::size_t local = 0; local < count; ++local) {
					options[slot].push_back(Labeled< int >{ label, local });
				}
			}
		}

		std::set< Lidx > result;
		for (const auto &slot_options : options) {
			if (slot_options.empty()) {
				return result;
			}
		}

		std::vector< std::size_t > pos(num_slots, 0);
		while (true) {
			Lidx idx(num_slots);
			for (std::size_t slot = 0; slot < num_slots; ++slot) {
				idx[slot] = options[slot][pos[slot]];
			}

			int product = identity;
			for (const auto &entry : idx) {
				product = combine(product, entry.label);
			}
			if (product == target && canonical(idx)) {
				result.insert(idx);
			}

			std::size_t slot = 0;
			for (; slot < num_slots; ++slot) {
				if (++pos[slot] < options[slot].size()) {
					break;
				}
				pos[slot] = 0;
			}
			if (slot == num_slots) {
				break;
			}
		}

		return result;
	}

	template< typename Combine >
	void check_layout(const SlotDims &dims, const util::PartitionList &parts, const std::vector< Symmetry > &signs,
					  Combine combine, int identity, int target) {
		const SectorLayout< int > layout(dims, parts, Constraint{ combine, identity, target }, signs);

		// The sector-major canonical tuples the layout ranks, and the flat (value-major) canonical tuples: these
		// pick different representatives per orbit but are equinumerous, so both independently pin down size().
		const std::set< Lidx > expected =
			brute_force(dims, combine, identity, target, [&](const Lidx &idx) { return layout.is_canonical(idx); });
		const std::set< Lidx > value_major = brute_force(
			dims, combine, identity, target, [&](const Lidx &idx) { return tpack::is_canonical(idx, parts, signs); });

		// size() matches a brute-force count of canonical, non-vanishing, constraint-satisfying tuples.
		ASSERT_EQ(layout.size(), expected.size());
		EXPECT_EQ(layout.size(), value_major.size());

		// unrank is a bijection onto exactly those tuples, every result is canonical and satisfies the rule, and
		// rank inverts it.
		std::set< Lidx > seen;
		for (std::size_t r = 0; r < layout.size(); ++r) {
			const Lidx idx = layout.unrank(r);
			EXPECT_TRUE(layout.is_canonical(idx));

			int product = identity;
			for (const auto &entry : idx) {
				product = combine(product, entry.label);
			}
			EXPECT_EQ(product, target);

			EXPECT_EQ(layout.rank(idx), r);
			seen.insert(idx);
		}
		EXPECT_EQ(seen, expected);

		// Every expected tuple ranks into range and round-trips.
		for (const Lidx &idx : expected) {
			const std::size_t r = layout.rank(idx);
			EXPECT_LT(r, layout.size());
			EXPECT_EQ(layout.unrank(r), idx);
		}

		// The sectors form a contiguous, non-empty cover of [0, size()).
		std::size_t running = 0;
		for (const auto &sector : layout.sectors()) {
			EXPECT_EQ(sector.offset, running);
			EXPECT_GT(sector.size, 0u);
			running += sector.size;
		}
		EXPECT_EQ(running, layout.size());
	}

	// XOR of irrep bitmasks is the group product for an abelian point group.
	constexpr auto xor_combine = [](int lhs, int rhs) { return lhs ^ rhs; };

	// A slot carrying labels 0..num_labels-1, each with the same number of local values.
	std::map< int, std::size_t > uniform_slot(int num_labels, std::size_t count) {
		std::map< int, std::size_t > slot;
		for (int label = 0; label < num_labels; ++label) {
			slot[label] = count;
		}
		return slot;
	}

	const std::vector< Symmetry > symmetric_one{ Symmetry::Symm };
	const std::vector< Symmetry > antisymmetric_one{ Symmetry::Antisymm };

} // namespace

TEST(TPack, SectorLayoutScalar) {
	// No indices: a single sector of one entry iff the empty product (identity) hits the target.
	check_layout({}, {}, {}, xor_combine, 0, 0);
	const SectorLayout< int > empty_target(SlotDims{}, util::PartitionList{}, Constraint{ xor_combine, 0, 1 },
										   std::vector< Symmetry >{});
	EXPECT_EQ(empty_target.size(), 0u);
}

TEST(TPack, SectorLayoutPairTwoLabels) {
	const SlotDims dims{ uniform_slot(2, 2), uniform_slot(2, 2) };
	const util::PartitionList parts{ { { 0, 1 } } };

	for (int target : { 0, 1 }) {
		check_layout(dims, parts, symmetric_one, xor_combine, 0, target);
		check_layout(dims, parts, antisymmetric_one, xor_combine, 0, target);
	}
}

TEST(TPack, SectorLayoutPairFourLabels) {
	const SlotDims dims{ uniform_slot(4, 3), uniform_slot(4, 3) };
	const util::PartitionList parts{ { { 0, 1 } } };

	for (int target : { 0, 1, 2, 3 }) {
		check_layout(dims, parts, symmetric_one, xor_combine, 0, target);
		check_layout(dims, parts, antisymmetric_one, xor_combine, 0, target);
	}
}

TEST(TPack, SectorLayoutPairEightLabels) {
	const SlotDims dims{ uniform_slot(8, 2), uniform_slot(8, 2) };
	const util::PartitionList parts{ { { 0, 1 } } };

	check_layout(dims, parts, symmetric_one, xor_combine, 0, 0);
	check_layout(dims, parts, antisymmetric_one, xor_combine, 0, 5);
}

TEST(TPack, SectorLayoutNonUniformLabelDims) {
	// Different labels carry different numbers of local values.
	std::map< int, std::size_t > slot{ { 0, 3 }, { 1, 1 }, { 2, 2 } };
	const SlotDims dims{ slot, slot };
	const util::PartitionList parts{ { { 0, 1 } } };

	check_layout(dims, parts, symmetric_one, xor_combine, 0, 0);
	check_layout(dims, parts, antisymmetric_one, xor_combine, 0, 0);
	check_layout(dims, parts, antisymmetric_one, xor_combine, 0, 3);
}

TEST(TPack, SectorLayoutZeroSizedLabels) {
	// Label 1 is listed but absent (count 0) at both slots; it must never appear in a sector.
	std::map< int, std::size_t > slot{ { 0, 2 }, { 1, 0 }, { 2, 2 }, { 3, 0 } };
	const SlotDims dims{ slot, slot };
	const util::PartitionList parts{ { { 0, 1 } } };

	check_layout(dims, parts, symmetric_one, xor_combine, 0, 0);
	check_layout(dims, parts, antisymmetric_one, xor_combine, 0, 2);
}

TEST(TPack, SectorLayoutMultiLevelPartition) {
	// A single partition with two levels and two columns: columns (0,2) and (1,3).
	const SlotDims dims{ uniform_slot(2, 2), uniform_slot(2, 2), uniform_slot(2, 2), uniform_slot(2, 2) };
	const util::PartitionList parts{ { { 0, 1 }, { 2, 3 } } };

	check_layout(dims, parts, symmetric_one, xor_combine, 0, 0);
	check_layout(dims, parts, antisymmetric_one, xor_combine, 0, 0);
	check_layout(dims, parts, symmetric_one, xor_combine, 0, 1);
}

TEST(TPack, SectorLayoutTwoIndependentPartitions) {
	const SlotDims dims{ uniform_slot(4, 2), uniform_slot(4, 2), uniform_slot(4, 2), uniform_slot(4, 2) };
	const util::PartitionList parts{ { { 0, 1 } }, { { 2, 3 } } };
	const std::vector< Symmetry > signs{ Symmetry::Symm, Symmetry::Antisymm };

	check_layout(dims, parts, signs, xor_combine, 0, 0);
	check_layout(dims, parts, signs, xor_combine, 0, 2);
}

TEST(TPack, SectorLayoutGenericEnumLabel) {
	enum class Irrep { A, B };
	using LidxE = std::vector< Labeled< Irrep > >;

	std::vector< std::map< Irrep, std::size_t > > dims{ { { Irrep::A, 2 }, { Irrep::B, 2 } },
														{ { Irrep::A, 2 }, { Irrep::B, 2 } } };
	const util::PartitionList parts{ { { 0, 1 } } };

	const SectorLayout< Irrep > layout(dims, parts, xor_constraint(Irrep::A, Irrep::A), symmetric_one);

	// Allowed canonical sectors (XOR == A): (A,A) and (B,B), each a symmetric pair of 2 locals -> C(3,2) = 3.
	ASSERT_EQ(layout.size(), 6u);
	for (std::size_t r = 0; r < layout.size(); ++r) {
		const LidxE idx = layout.unrank(r);
		EXPECT_TRUE(layout.is_canonical(idx));
		EXPECT_EQ(layout.rank(idx), r);
	}
}

} // namespace tpack::tests
