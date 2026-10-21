// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/details/canonical.hpp>
#include <tpack/details/symmetry.hpp>
#include <tpack/indexing.hpp>
#include <tpack/labeled.hpp>
#include <tpack/orbit.hpp>
#include <tpack/rank.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <map>
#include <ranges>
#include <utility>
#include <vector>

namespace tpack {

/**
 * @brief A group selection rule restricting which label tuples carry data.
 *
 * Models an abelian group product: the labels of all index positions are folded together with @c combine
 * (seeded with @c identity) and the tuple is allowed only when the result equals @c target. For a point group
 * @c combine is typically the XOR of irrep bitmasks, @c identity the totally symmetric irrep, and @c target
 * the symmetry of the stored quantity.
 *
 * @tparam Label The symmetry label type.
 * @tparam Combine A binary callable @c Label(Label, Label); must be associative and commutative (as a group
 *         product is) so the fold is independent of index order.
 */
template< typename Label, typename Combine > struct Constraint {
	Combine combine;
	Label identity;
	Label target;
};

/**
 * @brief Sector-graded packed layout for a tensor whose indices carry symmetry labels.
 *
 * Each index position (slot) takes values @c Labeled<Label>: a symmetry label plus a local index within that
 * label. A @e sector is an assignment of one label to every slot that is canonical under the ordinary
 * (symmetric) column ordering and allowed by the selection rule. Storage is the concatenation of the sectors,
 * each a contiguous block; within a sector the label of every slot is fixed, so what remains is an ordinary
 * integer packing problem over the local indices, which is delegated to num_orbits / rank / unrank.
 *
 * Antisymmetry acts only within a sector, on the local indices of equal-label column runs: a sector's labels
 * are always canonicalised symmetrically because two columns sharing a label but differing in local index are
 * distinct and non-vanishing; only equal @c (label, local) columns vanish.
 *
 * @tparam Label The symmetry label type; must be totally ordered (see Labeled).
 */
template< typename Label > class SectorLayout {
public:
	/// A partition as the collapsed-to-local packing sees it: a list of levels, each a list of slot positions.
	using Partition = std::vector< std::vector< std::size_t > >;

	/// A precomputed sector: its per-slot labels, its storage offset, and the number of entries it holds.
	struct Sector {
		std::vector< Label > labels;
		std::size_t offset;
		std::size_t size;
	};

	/**
	 * @brief Builds the layout and precomputes all allowed canonical sectors.
	 *
	 * @tparam Dims A range over slots; each slot is a range of @c {label, count} pairs giving the number of
	 *         local values for every label present at that slot (a @c std::map, a range of @c std::pair, ...).
	 *         Labels absent from a slot (or listed with count 0) simply do not occur there.
	 * @tparam Partitions A range of partitions describing the symmetry (same convention as the integer API).
	 * @tparam Combine The constraint's fold callable.
	 * @tparam Signs A single Symmetry, a per-partition range of Symmetry, or the all-symmetric sentinel.
	 * @param dims The per-slot label-to-count associations.
	 * @param partitions The symmetry partitions.
	 * @param constraint The group selection rule.
	 * @param signs The per-partition symmetry; defaults to all symmetric.
	 *
	 * @pre Within every level of a partition, all slots share an identical label-to-count association.
	 * @pre The partitions cover every slot that is to be ranked (as for the integer API; a free index is a
	 *      single-column partition).
	 */
	template< std::ranges::range Dims, std::ranges::range Partitions, typename Combine,
			  details::signs_spec Signs = details::all_symmetric_t >
	SectorLayout(const Dims &dims, const Partitions &partitions, const Constraint< Label, Combine > &constraint,
				 const Signs &signs = details::all_symmetric) {
		using std::ranges::begin;
		using std::ranges::end;

		// Materialise each slot's label -> count association, dropping absent (zero-count) labels and sorting by
		// label so the enumeration order and the canonical label ordering agree.
		for (auto &&slot : dims) {
			std::vector< std::pair< Label, std::size_t > > entries;
			for (auto &&[label, count] : slot) {
				if (count > 0) {
					entries.emplace_back(label, count);
				}
			}

			auto project_to_first = &decltype(entries)::value_type::first;

			std::ranges::sort(entries, {}, project_to_first);
			assert(std::ranges::adjacent_find(entries, {}, project_to_first) == entries.end());

			m_slot_dims.push_back(std::move(entries));
		}
		m_num_slots = m_slot_dims.size();

		// Materialise the partitions and their per-partition signs into owned storage.
		std::size_t part_index = 0;
		for (auto &&part : partitions) {
			Partition levels;
			for (auto &&level : part) {
				levels.emplace_back(begin(level), end(level));
			}

			m_partitions.push_back(std::move(levels));
			m_part_signs.push_back(details::antisymmetric_at(signs, part_index) ? Symmetry::Antisymm : Symmetry::Symm);
			++part_index;
		}

		assert(levels_uniform());

		enumerate_sectors(constraint);
	}

	/// @return The total number of stored entries, i.e. the size of the packed storage.
	std::size_t size() const { return m_size; }

	/// @return The precomputed sectors in storage (offset) order.
	const std::vector< Sector > &sectors() const { return m_sectors; }

	/**
	 * @brief Tests whether a labelled indexing is the canonical representative used by this layout.
	 *
	 * The canonical form is @e sector-major: within every partition the columns are non-increasing by label
	 * (compared reverse-lexicographically over levels), and columns sharing a label tuple are in addition
	 * non-increasing by local index (strictly decreasing for an antisymmetric partition, so equal columns
	 * vanish). Keeping the label the major key is what makes each sector a contiguous block. For single-level
	 * partitions this coincides with the free is_canonical; for multi-level partitions it differs, because the
	 * free function interleaves label and local per level whereas sector-graded storage must order by label
	 * first.
	 *
	 * @tparam Indexing A random-access range of @c Labeled<Label>.
	 * @param idx The labelled indexing to test.
	 * @return @c true if @p idx is sector-major canonical.
	 */
	template< std::ranges::random_access_range Indexing > bool is_canonical(const Indexing &idx) const {
		const auto it = std::ranges::begin(idx);
		auto label_of = [&](std::size_t slot) -> const Label & {
			return it[static_cast< std::ptrdiff_t >(slot)].label;
		};
		auto local_of = [&](std::size_t slot) { return it[static_cast< std::ptrdiff_t >(slot)].local; };

		// Label tuple first; only columns sharing it are ordered by their local indices.
		return details::is_canonical_by(
			m_partitions, m_part_signs, [&](const Partition &part, std::size_t lhs, std::size_t rhs) {
				const auto label_order = details::reverse_lex_compare(part, lhs, rhs, label_of);
				return label_order != 0 ? label_order : details::reverse_lex_compare(part, lhs, rhs, local_of);
			});
	}

	/**
	 * @brief Computes the packed rank of a canonical labelled indexing.
	 *
	 * @tparam Indexing A random-access range of @c Labeled<Label>.
	 * @param idx The canonical labelled indexing to rank.
	 * @return The rank in @c [0, size()): the owning sector's offset plus the in-sector rank.
	 *
	 * @pre @p idx is canonical (see is_canonical) and its label tuple is an allowed sector.
	 */
	template< std::ranges::random_access_range Indexing > std::size_t rank(const Indexing &idx) const {
		using std::ranges::size;
		assert(size(idx) == m_num_slots);
		assert(is_canonical(idx));

		const auto it = std::ranges::begin(idx);
		std::vector< Label > labels(m_num_slots);
		std::vector< std::size_t > locals(m_num_slots);
		for (std::size_t slot = 0; slot < m_num_slots; ++slot) {
			labels[slot] = it[static_cast< std::ptrdiff_t >(slot)].label;
			locals[slot] = it[static_cast< std::ptrdiff_t >(slot)].local;
		}

		const auto found = m_sector_index.find(labels);
		assert(found != m_sector_index.end());
		const std::size_t sector = found->second;
		const Block &block       = m_blocks[sector];

		return m_sectors[sector].offset + tpack::rank(locals, block.local_dims, block.run_parts, block.run_signs);
	}

	/**
	 * @brief Reconstructs the canonical labelled indexing from a rank (the inverse of rank()).
	 *
	 * @tparam Indexing A random-access range of @c Labeled<Label>.
	 * @param r The rank to decode; must lie in @c [0, size()).
	 * @param idx [out] Overwritten with the reconstructed canonical labelled indexing.
	 *
	 * @post is_canonical(idx, ...) holds.
	 */
	template< std::ranges::random_access_range Indexing > void unrank(std::size_t r, Indexing &idx) const {
		using std::ranges::size;
		assert(r < m_size);
		assert(size(idx) == m_num_slots);

		const std::size_t sector = sector_of(r);
		const Block &block       = m_blocks[sector];

		std::vector< std::size_t > locals(m_num_slots, 0);
		tpack::unrank(locals, r - m_sectors[sector].offset, block.local_dims, block.run_parts, block.run_signs);

		const auto it = std::ranges::begin(idx);
		for (std::size_t slot = 0; slot < m_num_slots; ++slot) {
			it[static_cast< std::ptrdiff_t >(slot)] = Labeled< Label >{ m_sectors[sector].labels[slot], locals[slot] };
		}

		assert(is_canonical(idx));
	}

	/**
	 * @brief Reconstructs and returns a freshly allocated canonical labelled indexing from a rank.
	 *
	 * @tparam Indexing The container to allocate and return (default @c std::vector<Labeled<Label>>).
	 * @param r The rank to decode; must lie in @c [0, size()).
	 * @return The reconstructed canonical labelled indexing, sized to the number of slots.
	 */
	template< std::ranges::random_access_range Indexing = std::vector< Labeled< Label > > >
	Indexing unrank(std::size_t r) const {
		Indexing idx;
		idx.resize(m_num_slots);
		unrank(r, idx);
		return idx;
	}

private:
	/// The in-sector integer packing problem: local dimensions plus the label-run-split partitions and signs.
	struct Block {
		std::vector< std::size_t > local_dims;
		std::vector< Partition > run_parts;
		std::vector< Symmetry > run_signs;
	};

	/// Looks up the number of local values for @p label at @p slot.
	std::size_t count_at(std::size_t slot, const Label &label) const {
		const auto &entries = m_slot_dims[slot];
		const auto it = std::ranges::lower_bound(entries, label, {}, [](const auto &entry) { return entry.first; });
		assert(it != entries.end() && it->first == label);
		return it->second;
	}

	/// Checks the per-level uniform-dimension precondition (intended for use inside @c assert).
	bool levels_uniform() const {
		for (const auto &part : m_partitions) {
			for (const auto &level : part) {
				if (level.empty()) {
					continue;
				}
				const auto &reference = m_slot_dims[level.front()];
				for (std::size_t slot : level) {
					if (m_slot_dims[slot] != reference) {
						return false;
					}
				}
			}
		}
		return true;
	}

	/// Whether columns @p lhs and @p rhs of @p part carry the same label on every level.
	bool same_label_column(const Partition &part, const std::vector< Label > &labels, std::size_t lhs,
						   std::size_t rhs) const {
		for (const auto &level : part) {
			if (!(labels[level[lhs]] == labels[level[rhs]])) {
				return false;
			}
		}
		return true;
	}

	/// Builds the in-sector integer problem for a fixed per-slot label assignment.
	Block build_block(const std::vector< Label > &labels) const {
		Block block;
		block.local_dims.resize(m_num_slots);
		for (std::size_t slot = 0; slot < m_num_slots; ++slot) {
			block.local_dims[slot] = count_at(slot, labels[slot]);
		}

		std::size_t part_index = 0;
		for (const auto &part : m_partitions) {
			const Symmetry sign        = m_part_signs[part_index++];
			const std::size_t num_cols = part.empty() ? 0 : part.front().size();

			// Equal-label columns are contiguous in a canonical sector, so split the partition into maximal runs
			// of equal label; each run is an ordinary sub-partition packed over the local indices.
			std::size_t run_start = 0;
			while (run_start < num_cols) {
				std::size_t run_end = run_start + 1;
				while (run_end < num_cols && same_label_column(part, labels, run_end - 1, run_end)) {
					++run_end;
				}

				Partition run;
				for (const auto &level : part) {
					run.emplace_back(level.begin() + static_cast< std::ptrdiff_t >(run_start),
									 level.begin() + static_cast< std::ptrdiff_t >(run_end));
				}
				block.run_parts.push_back(std::move(run));
				block.run_signs.push_back(sign);

				run_start = run_end;
			}
		}

		return block;
	}

	/// Enumerates every allowed canonical sector and records its offset and size.
	template< typename Combine > void enumerate_sectors(const Constraint< Label, Combine > &constraint) {
		// With a slot offering no label the product of choices is empty and there are no sectors.
		for (const auto &entries : m_slot_dims) {
			if (entries.empty()) {
				return;
			}
		}

		std::vector< std::size_t > pos(m_num_slots, 0);
		std::vector< Label > labels(m_num_slots);
		while (true) {
			for (std::size_t slot = 0; slot < m_num_slots; ++slot) {
				labels[slot] = m_slot_dims[slot][pos[slot]].first;
			}

			Label product = constraint.identity;
			for (std::size_t slot = 0; slot < m_num_slots; ++slot) {
				product = constraint.combine(product, labels[slot]);
			}

			if (product == constraint.target && tpack::is_canonical(labels, m_partitions, details::all_symmetric)) {
				Block block          = build_block(labels);
				const std::size_t sz = num_orbits(block.local_dims, block.run_parts, block.run_signs);
				if (sz > 0) {
					m_sector_index.emplace(labels, m_sectors.size());
					m_sectors.push_back(Sector{ labels, m_size, sz });
					m_blocks.push_back(std::move(block));
					m_size += sz;
				}
			}

			// Advance the mixed-radix odometer over the per-slot label choices (slot 0 varies fastest).
			std::size_t slot = 0;
			for (; slot < m_num_slots; ++slot) {
				if (++pos[slot] < m_slot_dims[slot].size()) {
					break;
				}
				pos[slot] = 0;
			}
			if (slot == m_num_slots) {
				break;
			}
		}
	}

	/// Returns the index of the sector owning rank @p r (the last sector whose offset does not exceed @p r).
	std::size_t sector_of(std::size_t r) const {
		std::size_t lo = 0;
		std::size_t hi = m_sectors.size();
		while (lo < hi) {
			const std::size_t mid = lo + (hi - lo) / 2;
			if (m_sectors[mid].offset <= r) {
				lo = mid + 1;
			} else {
				hi = mid;
			}
		}
		assert(lo > 0);
		return lo - 1;
	}

	std::size_t m_num_slots = 0;
	std::size_t m_size      = 0;
	std::vector< std::vector< std::pair< Label, std::size_t > > > m_slot_dims;
	std::vector< Partition > m_partitions;
	std::vector< Symmetry > m_part_signs;
	std::vector< Sector > m_sectors;
	std::vector< Block > m_blocks;
	std::map< std::vector< Label >, std::size_t > m_sector_index;
};

} // namespace tpack
