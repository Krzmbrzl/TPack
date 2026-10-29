// SPDX-License-Identifier: BSD-3-Clause

#include <tpack/labeled.hpp>
#include <tpack/sector_layout.hpp>
#include <tpack/sector_walker.hpp>
#include <tpack/symmetry.hpp>

#include <cstddef>
#include <map>
#include <vector>

#include <gtest/gtest.h>


namespace tpack::tests {

namespace {

	using SlotDims = std::vector< std::map< int, std::size_t > >;
	using Parts    = std::vector< std::vector< std::vector< std::size_t > > >;

	std::map< int, std::size_t > uniform_slot(int num_labels, std::size_t count) {
		std::map< int, std::size_t > slot;
		for (int label = 0; label < num_labels; ++label) {
			slot[label] = count;
		}
		return slot;
	}

	void check_walker(const SlotDims &dims, const Parts &parts, const std::vector< Symmetry > &signs, int target) {
		const SectorLayout< int > layout(dims, parts, xor_constraint(0, target), signs);
		ASSERT_GT(layout.size(), 0u);

		SectorWalker< int > walker(layout);
		for (std::size_t r = 0; r < layout.size(); ++r) {
			ASSERT_EQ(walker.rank(), r);
			ASSERT_EQ(walker.indexing(), layout.unrank(r)) << "at rank " << r;

			EXPECT_EQ(walker.next(), r + 1 < layout.size());
		}

		// Wrapped around to the start
		EXPECT_EQ(walker.rank(), 0u);
		EXPECT_EQ(walker.indexing(), layout.unrank(0));
	}

} // namespace

TEST(TPackSectorWalker, scalar) {
	check_walker({}, {}, {}, 0);
}

TEST(TPackSectorWalker, symmetric_pair) {
	check_walker({ uniform_slot(4, 3), uniform_slot(4, 3) }, { { { 0, 1 } } }, { Symmetry::Symm }, 0);
	check_walker({ uniform_slot(4, 3), uniform_slot(4, 3) }, { { { 0, 1 } } }, { Symmetry::Symm }, 3);
}

TEST(TPackSectorWalker, antisymmetric_pair) {
	check_walker({ uniform_slot(4, 3), uniform_slot(4, 3) }, { { { 0, 1 } } }, { Symmetry::Antisymm }, 0);
	check_walker({ uniform_slot(4, 3), uniform_slot(4, 3) }, { { { 0, 1 } } }, { Symmetry::Antisymm }, 2);
}

TEST(TPackSectorWalker, non_uniform_label_dims) {
	const std::map< int, std::size_t > slot{ { 0, 3 }, { 1, 1 }, { 2, 2 } };
	check_walker({ slot, slot }, { { { 0, 1 } } }, { Symmetry::Antisymm }, 0);
}

TEST(TPackSectorWalker, multi_level_partition) {
	const SlotDims dims(4, uniform_slot(2, 2));
	check_walker(dims, { { { 0, 1 }, { 2, 3 } } }, { Symmetry::Symm }, 0);
	check_walker(dims, { { { 0, 1 }, { 2, 3 } } }, { Symmetry::Antisymm }, 0);
}

TEST(TPackSectorWalker, two_partitions) {
	const SlotDims dims(4, uniform_slot(4, 2));
	check_walker(dims, { { { 0, 1 } }, { { 2, 3 } } }, { Symmetry::Symm, Symmetry::Antisymm }, 1);
}

} // namespace tpack::tests
