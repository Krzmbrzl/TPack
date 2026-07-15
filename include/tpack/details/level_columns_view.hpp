// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cassert>
#include <compare>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <utility>
#include <vector>

namespace tpack::details {

template< typename Proxy, std::ranges::range Levels > class LevelColumnsViewBase {
public:
	// See also https://artificial-mind.net/blog/2020/11/28/std-sort-multiple-ranges
	friend struct ColVal;
	struct ColVal {
		std::vector< typename Proxy::value_type > values;

		friend constexpr auto operator<=>(const ColVal &lhs, const ColVal &rhs) {
			const std::size_t size = lhs.values.size();

			// Need to compare in reverse order to stay consistent with ColRef impl
			for (std::size_t i = 0; i < size; ++i) {
				if (lhs.values[size - i - 1] != rhs.values[size - i - 1]) {
					return lhs.values[size - i - 1] <=> rhs.values[size - i - 1];
				}
			}

			return decltype(std::declval< typename Proxy::value_type >()
							<=> std::declval< typename Proxy::value_type >())::equivalent;
		}

		friend constexpr bool operator==(const ColVal &lhs, const ColVal &rhs) = default;
	};
	static_assert(std::totally_ordered< ColVal >);

	friend struct ColRef;
	struct ColRef {
		constexpr ColRef(std::size_t col, LevelColumnsViewBase &view) : m_col(col), m_col_view(view) {}

		constexpr ColRef &operator=(ColRef &&other) {
			for (auto &level : m_col_view.levels()) {
				level[m_col] = std::move(m_col_view.proxy()[level[other.m_col]]);
			}

			return *this;
		}
		constexpr ColRef &operator=(ColRef &&other) const {
			for (auto &level : m_col_view.levels()) {
				level[m_col] = std::move(m_col_view.proxy()[level[other.m_col]]);
			}

			return *this;
		}

		constexpr ColRef &operator=(ColVal &&val) {
			std::size_t i = 0;
			for (auto &level : m_col_view.levels()) {
				m_col_view.proxy()[level[m_col]] = std::move(val.values[i]);
				++i;
			}

			return *this;
		}
		constexpr ColRef &operator=(ColVal &&val) const {
			std::size_t i = 0;
			for (const auto &level : m_col_view.levels()) {
				m_col_view.proxy()[level[m_col]] = std::move(val.values[i]);
				++i;
			}

			return *this;
		}

		constexpr operator ColVal() const & {
			using std::ranges::size;

			ColVal val{};
			val.values.reserve(size(m_col_view.levels()));

			for (const auto &level : m_col_view.levels()) {
				val.values.emplace_back(std::move(m_col_view.proxy()[level[m_col]]));
			}

			return val;
		}

		constexpr operator ColVal() && {
			using std::ranges::size;

			ColVal val{};
			val.values.reserve(size(m_col_view.levels()));

			for (const auto &level : m_col_view.levels()) {
				val.values.emplace_back(std::move(m_col_view.proxy()[level[m_col]]));
			}

			return val;
		}

		friend constexpr void swap(ColRef lhs, ColRef rhs) {
			using std::swap;
			for (auto &level : lhs.m_col_view.levels()) {
				swap(lhs.m_col_view.proxy()[level[lhs.m_col]], rhs.m_col_view.proxy()[level[rhs.m_col]]);
			}
		}

		friend constexpr auto operator<=>(const ColRef &lhs, const ColRef &rhs) {
			for (const auto &level : std::ranges::views::reverse(lhs.m_col_view.levels())) {
				const auto &lhs_val = lhs.m_col_view.proxy()[level[lhs.m_col]];
				const auto &rhs_val = rhs.m_col_view.proxy()[level[rhs.m_col]];
				if (lhs_val != rhs_val) {
					return lhs_val <=> rhs_val;
				}
			}

			return decltype(std::declval< typename Proxy::value_type >()
							<=> std::declval< typename Proxy::value_type >())::equivalent;
		}
		friend constexpr auto operator<=>(const ColVal &lhs, const ColRef &rhs) {
			std::size_t i = lhs.values.size() - 1;
			for (const auto &level : std::ranges::views::reverse(rhs.m_col_view.levels())) {
				const auto &rhs_val = rhs.m_col_view.proxy()[level[rhs.m_col]];
				if (lhs.values[i] != rhs_val) {
					return lhs.values[i] <=> rhs_val;
				}
				--i;
			}

			return decltype(std::declval< typename Proxy::value_type >()
							<=> std::declval< typename Proxy::value_type >())::equivalent;
		}
		friend constexpr auto operator<=>(const ColRef &lhs, const ColVal &rhs) {
			// Inverts less_than or greater_than result of rhs <=> lhs
			return 0 <=> (rhs <=> lhs);
		}
		friend constexpr bool operator==(const ColRef &lhs, const ColRef &rhs) { return (lhs <=> rhs) == 0; }
		friend constexpr bool operator==(const ColVal &lhs, const ColRef &rhs) { return (lhs <=> rhs) == 0; }
		friend constexpr bool operator==(const ColRef &lhs, const ColVal &rhs) { return (lhs <=> rhs) == 0; }

		std::size_t m_col;
		LevelColumnsViewBase &m_col_view;
	};

	friend struct ColIter;
	struct ColIter {
		using difference_type   = std::ptrdiff_t;
		using value_type        = ColVal;
		using reference         = ColRef;
		using iterator_category = std::random_access_iterator_tag;


		std::size_t m_col;
		LevelColumnsViewBase *m_col_view;

		constexpr std::strong_ordering operator<=>(const ColIter &other) const { return m_col <=> other.m_col; }

		constexpr difference_type operator-(const ColIter &other) const {
			return static_cast< difference_type >(m_col - other.m_col);
		}
		constexpr difference_type operator+(const ColIter &other) const {
			return static_cast< difference_type >(m_col + other.m_col);
		}

		constexpr ColIter &operator+=(difference_type amount) {
			assert(amount >= 0 || static_cast< std::size_t >(-amount) >= m_col);
			if (amount >= 0) {
				m_col += static_cast< std::size_t >(amount);
			} else {
				m_col -= static_cast< std::size_t >(-amount);
			}
			return *this;
		}
		constexpr ColIter &operator-=(difference_type amount) {
			assert(amount >= 0 || static_cast< std::size_t >(-amount) >= m_col);
			if (amount >= 0) {
				m_col -= static_cast< std::size_t >(amount);
			} else {
				m_col += static_cast< std::size_t >(-amount);
			}
			return *this;
		}
		friend constexpr ColIter operator+(const ColIter &iter, difference_type amount) {
			ColIter copy = iter;
			copy += amount;
			return copy;
		}
		friend constexpr ColIter operator+(difference_type amount, const ColIter &iter) { return iter + amount; }
		friend constexpr ColIter operator-(const ColIter &iter, difference_type amount) {
			ColIter copy = iter;
			copy -= amount;
			return copy;
		}
		friend constexpr ColIter operator-(difference_type amount, const ColIter &iter) { return iter - amount; }

		constexpr ColIter &operator++() { return *this += 1; }
		constexpr ColIter operator++(int) {
			ColIter copy = *this;
			++(*this);
			return copy;
		}

		constexpr ColIter &operator--() { return *this -= 1; }
		constexpr ColIter operator--(int) {
			ColIter copy = *this;
			--(*this);
			return copy;
		}

		constexpr reference operator[](difference_type idx) const { return { m_col + idx, *m_col_view }; }

		constexpr bool operator==(const ColIter &other) const { return m_col == other.m_col; }

		constexpr reference operator*() { return { m_col, *m_col_view }; }
		constexpr reference operator*() const { return { m_col, *m_col_view }; }
	};

	friend constexpr void iter_swap(ColIter lhs, ColIter rhs) { swap(*lhs, *rhs); }

	static_assert(std::bidirectional_iterator< ColIter >);
	static_assert(std::random_access_iterator< ColIter >);


	using iterator = ColIter;

	constexpr LevelColumnsViewBase(Levels &levels, Proxy &&proxy)
		: m_levels(levels), m_proxy(std::forward< Proxy >(proxy)) {}

	constexpr std::size_t num_cols() const {
		using std::ranges::begin;
		using std::ranges::size;
		return size(*begin(m_levels));
	}

	constexpr iterator begin() { return { 0, this }; }
	constexpr iterator end() { return { num_cols(), this }; }

	constexpr ColRef operator[](std::size_t col) { return { col, *this }; }

	constexpr Levels &levels() { return m_levels; }
	constexpr const Levels &levels() const { return m_levels; }

	constexpr Proxy &proxy() { return m_proxy; }
	constexpr const Proxy &proxy() const { return m_proxy; }

private:
	Levels &m_levels;
	Proxy m_proxy;
};

template< std::ranges::random_access_range Indexing > struct IndexingProxy {
	using value_type      = std::ranges::range_value_t< Indexing >;
	using reference       = std::add_lvalue_reference_t< value_type >;
	using const_reference = std::add_lvalue_reference_t< std::add_const_t< value_type > >;

	constexpr IndexingProxy(Indexing &indexing) : m_indexing(indexing) {}

	constexpr std::weak_ordering cmp(std::size_t lhs_idx, std::size_t rhs_idx) const {
		return m_indexing[lhs_idx] <=> m_indexing[rhs_idx];
	}

	constexpr void swap(std::size_t lhs_idx, std::size_t rhs_idx) {
		using std::swap;
		swap(m_indexing[lhs_idx], m_indexing[rhs_idx]);
	}

	constexpr reference operator[](std::size_t idx) { return m_indexing[idx]; }

	constexpr const_reference operator[](std::size_t idx) const { return m_indexing[idx]; }

	Indexing &m_indexing;
};

template< std::ranges::range Levels > struct LevelProxy {
	using value_type      = std::ranges::range_value_t< std::ranges::range_value_t< Levels > >;
	using reference       = std::add_lvalue_reference_t< value_type >;
	using const_reference = std::add_lvalue_reference_t< std::add_const_t< value_type > >;

	constexpr std::weak_ordering cmp(std::size_t lhs_idx, std::size_t rhs_idx) const { return lhs_idx <=> rhs_idx; }

	constexpr void swap(reference lhs_idx, reference rhs_idx) {
		using std::swap;
		swap(lhs_idx, rhs_idx);
	}

	constexpr reference operator[](reference idx) { return idx; }

	constexpr const_reference operator[](const_reference idx) const { return idx; }
};

template< std::ranges::range Levels, std::ranges::random_access_range Indexing >
struct LevelColumnsIndexingView : LevelColumnsViewBase< IndexingProxy< Indexing >, Levels > {
	LevelColumnsIndexingView(Levels &levels, Indexing &idx)
		: LevelColumnsViewBase< IndexingProxy< Indexing >, Levels >(levels, IndexingProxy(idx)) {}
};

template< std::ranges::range Levels > struct LevelColumnsView : LevelColumnsViewBase< LevelProxy< Levels >, Levels > {
	LevelColumnsView(Levels &levels)
		: LevelColumnsViewBase< LevelProxy< Levels >, Levels >(levels, LevelProxy< Levels >{}) {}
};


static_assert(std::totally_ordered< LevelColumnsView< std::vector< std::vector< std::size_t > > >::ColRef >);
static_assert(std::totally_ordered_with< LevelColumnsView< std::vector< std::vector< std::size_t > > >::ColRef,
										 LevelColumnsView< std::vector< std::vector< std::size_t > > >::ColVal >);

} // namespace tpack::details
