// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/concepts.hpp>
#include <tpack/details/levels.hpp>
#include <tpack/details/symmetry.hpp>

#include <cstddef>
#include <ranges>
#include <vector>

namespace tpack {

/**
 * @brief Stateful helper that runs through all canonical indexings in order of their rank.
 *
 * Calling unrank() for consecutive ranks repeats a greedy inversion of the combinatorial number system for every
 * rank. Since consecutive ranks differ only in a few trailing columns, this helper instead keeps the current
 * indexing and advances it in place. Starting at rank 0, it visits exactly the indexings unrank() yields for
 * rank 0, 1, 2, ... without ever inverting the binomial formula.
 *
 * The helper takes a copy of everything it needs, so the dimensions and partitions passed to the constructor
 * need not outlive it.
 *
 * @see unrank, num_orbits
 */
class RankWalker {
public:
	/**
	 * @brief Creates a walker positioned at rank 0.
	 *
	 * @tparam Dimensions A random-access range mapping each index to its dimension.
	 * @tparam Partitions A range of partitions describing the symmetry.
	 * @tparam Signs A single Symmetry (applied to every partition), a random-access range of per-partition
	 *         Symmetry flags, or the all_symmetric_t sentinel.
	 * @param dims The dimension of every index.
	 * @param partitions The symmetry partitions.
	 * @param signs A single Symmetry applied to every partition, or one Symmetry per partition; defaults to all
	 *        partitions symmetric.
	 *
	 * @pre @c num_orbits(dims, partitions, signs) > 0.
	 * @pre Within every level of a partition, all indices share the same dimension.
	 */
	template< dimensions Dimensions, partition_list Partitions, partition_signs Signs = all_symmetric_t >
	RankWalker(Dimensions &&dims, Partitions &&partitions, const Signs &signs = all_symmetric) {
		assign(dims, partitions, signs);
	}

	/**
	 * @brief Re-targets the walker to a new problem and positions it at rank 0.
	 *
	 * Equivalent to constructing a new walker, but reuses the storage of this one.
	 *
	 * @see RankWalker(Dimensions &&, Partitions &&, const Signs &) for the parameters and preconditions.
	 */
	template< dimensions Dimensions, partition_list Partitions, partition_signs Signs = all_symmetric_t >
	void assign(const Dimensions &dims, const Partitions &partitions, const Signs &signs = all_symmetric) {
		details::assert_signs_match(signs, partitions);

		m_indexing.assign(std::ranges::size(dims), 0);
		m_rank = 0;
		m_parts.resize(std::ranges::size(partitions));

		std::size_t part_index = 0;
		for (auto &&part_levels : partitions) {
			Part &part = m_parts[part_index];

			part.antisym = details::antisymmetric_at(signs, part_index);
			++part_index;

			part.num_cols = details::num_columns(part_levels);
			part.dim      = details::effective_dim(dims, part_levels);
			details::assign_level_strides(dims, part_levels, part.strides);

			part.levels.resize(std::ranges::size(part_levels));
			std::size_t level_index = 0;
			for (auto &&level : part_levels) {
				part.levels[level_index++].assign(std::ranges::begin(level), std::ranges::end(level));
			}

			part.effective.resize(part.num_cols);
			reset(part);
		}
	}

	/**
	 * @brief Advances to the next rank.
	 *
	 * @return @c true if the walker now sits at the next rank. @c false if it was at the last rank, in which case
	 *         it wraps around to rank 0.
	 */
	bool next() {
		for (Part &part : m_parts) {
			if (advance(part)) {
				++m_rank;
				return true;
			}
		}

		m_rank = 0;
		return false;
	}

	/// @return The rank of the current indexing.
	std::size_t rank() const { return m_rank; }

	/// @return The canonical indexing for the current rank(), i.e. what unrank() returns for it.
	const std::vector< std::size_t > &indexing() const { return m_indexing; }

private:
	struct Part {
		bool antisym         = false;
		std::size_t num_cols = 0;
		std::size_t dim      = 0;
		std::vector< std::vector< std::size_t > > levels;
		std::vector< std::size_t > strides;
		std::vector< std::size_t > effective;
	};

	std::vector< std::size_t > m_indexing;
	std::vector< Part > m_parts;
	std::size_t m_rank = 0;

	/// Sets all columns from @p first on to the smallest values allowed after the preceding ones and writes them
	/// into the indexing.
	void fill_from(Part &part, std::size_t first) {
		for (std::size_t col = first; col < part.num_cols; ++col) {
			// A strictly decreasing run bottoms out at (num_cols - 1 - col); a non-increasing one at 0.
			part.effective[col] = part.antisym ? part.num_cols - 1 - col : 0;
			details::split_column(part.levels, part.strides, col, part.effective[col], m_indexing);
		}
	}

	void reset(Part &part) { fill_from(part, 0); }

	/// Moves @p part to its next rank; if it was already at its last one, resets it to its first and returns @c false.
	bool advance(Part &part) {
		// The rightmost column that can grow while keeping the columns canonical: the first column is bounded by the
		// dimension, every other one by its left neighbour (strictly for an antisymmetric partition).
		for (std::size_t col = part.num_cols; col-- > 0;) {
			const std::size_t bound = col == 0 ? part.dim : part.effective[col - 1] + (part.antisym ? 0 : 1);
			if (part.effective[col] + 1 < bound) {
				++part.effective[col];
				details::split_column(part.levels, part.strides, col, part.effective[col], m_indexing);
				fill_from(part, col + 1);
				return true;
			}
		}

		reset(part);
		return false;
	}
};

} // namespace tpack
