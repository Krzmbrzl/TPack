// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/concepts.hpp>
#include <tpack/details/canonical.hpp>
#include <tpack/details/levels.hpp>
#include <tpack/details/symmetry.hpp>
#include <tpack/indexing.hpp>
#include <tpack/labeled.hpp>
#include <tpack/orbit.hpp>
#include <tpack/rank.hpp>

#include <algorithm>
#include <cassert>
#include <compare>
#include <cstddef>
#include <map>
#include <numeric>
#include <ranges>
#include <type_traits>
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
 * @brief Builds a Constraint whose group product is the bitwise XOR of the labels.
 *
 * This is the common case for abelian point groups, where irreps are XOR-combined bitmasks and the totally
 * symmetric irrep is the all-zero mask. Works for an integral or an enumeration @c Label.
 *
 * @tparam Label The symmetry label type (integral or enum).
 * @param identity The totally symmetric label, i.e. the fold's seed (usually all-zero).
 * @param target The required combined label of the stored quantity.
 * @return A Constraint using bitwise XOR as its @c combine.
 */
template< label Label > constexpr auto xor_constraint(Label identity, Label target) {
	constexpr auto combine = [](Label lhs, Label rhs) {
		if constexpr (std::is_enum_v< Label >) {
			using U = std::underlying_type_t< Label >;
			return static_cast< Label >(static_cast< U >(lhs) ^ static_cast< U >(rhs));
		} else {
			return static_cast< Label >(lhs ^ rhs);
		}
	};
	return Constraint{ combine, identity, target };
}

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
template< label Label > class SectorLayout {
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
	template< labeled_dimensions< Label > Dims, partition_list Partitions, label_combiner< Label > Combine,
			  partition_signs Signs = all_symmetric_t >
	SectorLayout(const Dims &dims, const Partitions &partitions, const Constraint< Label, Combine > &constraint,
				 const Signs &signs = all_symmetric) {
		using std::ranges::begin;
		using std::ranges::end;

		// Materialise each slot's label -> count association, dropping absent (zero-count) labels and sorting by
		// label so the enumeration order and the canonical label ordering agree.
		for (auto &&slot : dims) {
			std::vector< LabeledSlotDim > entries;
			for (auto &&[label, count] : slot) {
				if (count > 0) {
					entries.emplace_back(label, count);
				}
			}

			std::ranges::sort(entries, {}, &LabeledSlotDim::label);
			assert(std::ranges::adjacent_find(entries, {}, &LabeledSlotDim::label) == entries.end());

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
	 * @param indexing The labelled indexing to test.
	 * @return @c true if @p indexing is sector-major canonical.
	 */
	template< labeled_indexing< Label > Indexing > bool is_canonical(const Indexing &indexing) const {
		const auto it = std::ranges::begin(indexing);
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
	 * @param indexing The canonical labelled indexing to rank.
	 * @return The rank in @c [0, size()): the owning sector's offset plus the in-sector rank.
	 *
	 * @pre @p indexing is canonical (see is_canonical) and its label tuple is an allowed sector.
	 */
	template< labeled_indexing< Label > Indexing > std::size_t rank(const Indexing &indexing) const {
		using std::ranges::size;
		assert(size(indexing) == m_num_slots);
		assert(is_canonical(indexing));

		std::vector< Label > labels(m_num_slots);
		std::vector< std::size_t > locals(m_num_slots);
		std::ranges::transform(indexing, labels.begin(), &Labeled< Label >::label);
		std::ranges::transform(indexing, locals.begin(), &Labeled< Label >::local);

		const auto found = m_sector_index.find(labels);
		assert(found != m_sector_index.end());
		const std::size_t sector = found->second;
		const SectorBlock &block = m_blocks[sector];

		return m_sectors[sector].offset + tpack::rank(locals, block.local_dims, block.run_parts, block.run_signs);
	}

	/**
	 * @brief Reconstructs the canonical labelled indexing from a rank (the inverse of rank()).
	 *
	 * @tparam Indexing A random-access range of @c Labeled<Label>.
	 * @param rank The rank to decode; must lie in @c [0, size()).
	 * @param indexing [out] Overwritten with the reconstructed canonical labelled indexing.
	 *
	 * @post is_canonical(indexing, ...) holds.
	 */
	template< writable_labeled_indexing< Label > Indexing > void unrank(std::size_t rank, Indexing &indexing) const {
		using std::ranges::size;
		assert(rank < m_size);
		assert(size(indexing) == m_num_slots);

		const std::size_t sector = sector_of(rank);
		const SectorBlock &block = m_blocks[sector];

		std::vector< std::size_t > locals(m_num_slots, 0);
		tpack::unrank(rank - m_sectors[sector].offset, locals, block.local_dims, block.run_parts, block.run_signs);

		std::ranges::transform(m_sectors[sector].labels, locals, std::ranges::begin(indexing),
							   [](const Label &label, std::size_t local) {
								   return Labeled< Label >{ label, local };
							   });

		assert(is_canonical(indexing));
	}

	/**
	 * @brief Reconstructs and returns a freshly allocated canonical labelled indexing from a rank.
	 *
	 * @tparam Indexing The container to allocate and return (default @c std::vector<Labeled<Label>>).
	 * @param rank The rank to decode; must lie in @c [0, size()).
	 * @return The reconstructed canonical labelled indexing, sized to the number of slots.
	 */
	template< writable_labeled_indexing< Label > Indexing = std::vector< Labeled< Label > > >
	Indexing unrank(std::size_t rank) const {
		Indexing indexing;
		indexing.resize(m_num_slots);
		unrank(rank, indexing);
		return indexing;
	}

private:
	/// The in-sector integer packing problem for one Sector: local dimensions plus the label-run-split
	/// partitions and signs. Paired with the public Sector record at the same index in m_sectors / m_blocks.
	struct SectorBlock {
		std::vector< std::size_t > local_dims;
		std::vector< Partition > run_parts;
		std::vector< Symmetry > run_signs;
	};
	struct LabeledSlotDim {
		Label label;
		std::size_t count;

		auto operator<=>(const LabeledSlotDim &) const = default;
	};

	std::size_t m_num_slots = 0;
	std::size_t m_size      = 0;
	std::vector< std::vector< LabeledSlotDim > > m_slot_dims;
	std::vector< Partition > m_partitions;
	std::vector< Symmetry > m_part_signs;
	std::vector< Sector > m_sectors;
	std::vector< SectorBlock > m_blocks;
	std::map< std::vector< Label >, std::size_t > m_sector_index;


	/// Looks up the number of local values for @p label at @p slot.
	std::size_t count_at(std::size_t slot, const Label &label) const {
		const auto &entries = m_slot_dims[slot];
		const auto it       = std::ranges::lower_bound(entries, label, {}, &LabeledSlotDim::label);
		assert(it != entries.end() && it->label == label);
		return it->count;
	}

	/// Checks the per-level uniform-dimension precondition (intended for use inside @c assert).
	bool levels_uniform() const {
		return std::ranges::all_of(m_partitions, [&](const Partition &part) {
			return std::ranges::all_of(part, [&](const auto &level) {
				return std::ranges::all_of(
					level, [&](std::size_t slot) { return m_slot_dims[slot] == m_slot_dims[level.front()]; });
			});
		});
	}

	/// Whether columns @p lhs and @p rhs of @p part carry the same label on every level.
	bool same_label_column(const Partition &part, const std::vector< Label > &labels, std::size_t lhs,
						   std::size_t rhs) const {
		return details::reverse_lex_compare(part, lhs, rhs,
											[&labels](std::size_t slot) -> const Label & { return labels[slot]; })
			   == 0;
	}

	/// Builds the in-sector integer problem for a fixed per-slot label assignment.
	SectorBlock build_block(const std::vector< Label > &labels) const {
		SectorBlock block;
		block.local_dims.resize(m_num_slots);
		for (std::size_t slot = 0; slot < m_num_slots; ++slot) {
			block.local_dims[slot] = count_at(slot, labels[slot]);
		}

		std::size_t part_index = 0;
		for (const auto &part : m_partitions) {
			const Symmetry sign        = m_part_signs[part_index++];
			const std::size_t num_cols = details::num_columns(part);

			// Equal-label columns are contiguous in a canonical sector, so split the partition into maximal runs
			// of equal label; each run is an ordinary sub-partition packed over the local indices.
			const auto columns    = std::views::iota(std::size_t{ 0 }, num_cols);
			std::size_t run_start = 0;
			while (run_start < num_cols) {
				const auto remaining = columns | std::views::drop(run_start);
				const auto boundary  = std::ranges::adjacent_find(remaining, [&](std::size_t lhs, std::size_t rhs) {
                    return !same_label_column(part, labels, lhs, rhs);
                });
				const std::size_t run_end = boundary == remaining.end() ? num_cols : *boundary + 1;

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
		if (std::ranges::any_of(m_slot_dims, [](const auto &entries) { return entries.empty(); })) {
			return;
		}

		std::vector< std::size_t > pos(m_num_slots, 0);
		std::vector< Label > labels(m_num_slots);
		while (true) {
			for (std::size_t slot = 0; slot < m_num_slots; ++slot) {
				labels[slot] = m_slot_dims[slot][pos[slot]].label;
			}

			Label product = std::accumulate(labels.begin(), labels.end(), constraint.identity, constraint.combine);

			if (product == constraint.target && tpack::is_canonical(labels, m_partitions, all_symmetric)) {
				SectorBlock block    = build_block(labels);
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
		const auto next = std::ranges::upper_bound(m_sectors, r, {}, &Sector::offset);
		assert(next != m_sectors.begin());
		return static_cast< std::size_t >(next - m_sectors.begin()) - 1;
	}
};

} // namespace tpack
