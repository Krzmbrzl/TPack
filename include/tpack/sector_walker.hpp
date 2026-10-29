// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/concepts.hpp>
#include <tpack/labeled.hpp>
#include <tpack/rank_walker.hpp>
#include <tpack/sector_layout.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

namespace tpack {

/**
 * @brief Stateful helper that runs through all canonical labelled indexings of a SectorLayout in order of their rank.
 *
 * The labelled counterpart of RankWalker: starting at rank 0, it visits exactly the indexings
 * SectorLayout::unrank yields for rank 0, 1, 2, ... but advances the current indexing in place. Within a sector it
 * delegates to a RankWalker over the local indices; the labels only change when crossing into the next sector.
 *
 * The walker refers to the layout it was created from, which must outlive it.
 *
 * @tparam Label The symmetry label type of the layout.
 *
 * @see RankWalker, SectorLayout
 */
template< label Label > class SectorWalker {
public:
	/**
	 * @brief Creates a walker positioned at rank 0.
	 *
	 * @param layout The layout to walk.
	 *
	 * @pre @c layout.size() > 0.
	 */
	explicit SectorWalker(const SectorLayout< Label > &layout)
		: m_layout(&layout), m_indexing(layout.m_num_slots), m_walker(first_walker(layout)) {
		enter_sector(0);
	}

	/**
	 * @brief Advances to the next rank.
	 *
	 * @return @c true if the walker now sits at the next rank. @c false if it was at the last rank, in which case
	 *         it wraps around to rank 0.
	 */
	bool next() {
		if (m_walker.next()) {
			++m_rank;
			copy_locals();
			return true;
		}

		if (m_sector + 1 < m_layout->m_sectors.size()) {
			++m_rank;
			enter_sector(m_sector + 1);
			return true;
		}

		m_rank = 0;
		enter_sector(0);
		return false;
	}

	/// @return The rank of the current indexing.
	std::size_t rank() const { return m_rank; }

	/// @return The canonical labelled indexing for the current rank(), i.e. what SectorLayout::unrank returns for it.
	const std::vector< Labeled< Label > > &indexing() const { return m_indexing; }

private:
	const SectorLayout< Label > *m_layout;
	std::vector< Labeled< Label > > m_indexing;
	RankWalker m_walker;
	std::size_t m_sector = 0;
	std::size_t m_rank   = 0;

	static RankWalker first_walker(const SectorLayout< Label > &layout) {
		assert(layout.size() > 0);
		const auto &block = layout.m_blocks.front();
		return RankWalker(block.local_dims, block.run_parts, block.run_signs);
	}

	void enter_sector(std::size_t sector) {
		const auto &block = m_layout->m_blocks[sector];

		m_sector = sector;
		m_walker.assign(block.local_dims, block.run_parts, block.run_signs);
		std::ranges::transform(m_layout->m_sectors[sector].labels, m_indexing.begin(), [](const Label &label) {
			return Labeled< Label >{ label, 0 };
		});
		copy_locals();
	}

	void copy_locals() {
		std::ranges::transform(m_indexing, m_walker.indexing(), m_indexing.begin(),
							   [](Labeled< Label > entry, std::size_t local) {
								   entry.local = local;
								   return entry;
							   });
	}
};

} // namespace tpack
