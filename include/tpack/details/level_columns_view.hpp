// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstddef>
#include <iterator>
#include <ranges>
#include <utility>

namespace tpack::details {

template< std::ranges::range Levels, std::ranges::random_access_range Indexing > class LevelColumnsView {
public:
	friend struct Column;
	struct Column {
		std::size_t m_col;
		LevelColumnsView &m_col_view;

		constexpr bool operator<(const Column &other) const {
			for (auto &&level : std::ranges::views::reverse(m_col_view.m_levels)) {
				std::size_t lhs_idx = level[m_col];
				std::size_t rhs_idx = level[other.m_col];

				if (m_col_view.m_indexing[lhs_idx] > m_col_view.m_indexing[rhs_idx]) {
					return true;
				}
				if (m_col_view.m_indexing[lhs_idx] < m_col_view.m_indexing[rhs_idx]) {
					return false;
				}
			}

			// They are equal
			return false;
		}
	};

	struct ColumnIter {
		using difference_type   = std::ptrdiff_t;
		using element_type      = Column;
		using iterator_category = std::bidirectional_iterator_tag;


		std::size_t m_col;
		LevelColumnsView *m_col_view;

		constexpr ColumnIter &operator++() {
			m_col++;
			return *this;
		}
		constexpr ColumnIter operator++(int) {
			ColumnIter copy = *this;
			++(*this);
			return copy;
		}

		constexpr ColumnIter &operator--() {
			m_col--;
			return *this;
		}
		constexpr ColumnIter operator--(int) {
			ColumnIter copy = *this;
			--(*this);
			return copy;
		}

		constexpr bool operator==(const ColumnIter &other) const { return m_col == other.m_col; }

		constexpr element_type operator*() { return { m_col, *m_col_view }; }
		constexpr element_type operator*() const { return { m_col, *m_col_view }; }
	};

	friend void swap(Column lhs, Column rhs) { lhs.m_col_view.swap_cols(lhs.m_col, rhs.m_col); }

	friend void iter_swap(ColumnIter lhs, ColumnIter rhs) {
		using std::swap;
		swap(*lhs, *rhs);
	}

	static_assert(std::bidirectional_iterator< ColumnIter >);
	static_assert(std::same_as< std::bidirectional_iterator_tag,
								typename std::iterator_traits< ColumnIter >::iterator_category >);


	using iterator = ColumnIter;

	LevelColumnsView(Levels &levels, Indexing &idx) : m_levels(levels), m_indexing(idx) {}

	constexpr std::size_t num_cols() const {
		using std::ranges::begin;
		using std::ranges::size;
		return size(*begin(m_levels));
	}

	constexpr iterator begin() { return { 0, this }; }
	constexpr iterator end() { return { num_cols(), this }; }

	constexpr void swap_cols(std::size_t lhs, std::size_t rhs) {
		for (auto &&level : m_levels) {
			std::size_t lhs_idx = level[lhs];
			std::size_t rhs_idx = level[rhs];

			using std::swap;
			swap(m_indexing[lhs_idx], m_indexing[rhs_idx]);
		}
	}

private:
	Levels &m_levels;
	Indexing &m_indexing;
};

} // namespace tpack::details
