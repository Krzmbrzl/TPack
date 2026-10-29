// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <tpack/labeled.hpp>
#include <tpack/symmetry.hpp>

#include <compare>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <tuple>
#include <type_traits>

namespace tpack {

/**
 * @brief A level of a partition: a list of tensor index positions.
 *
 * Positions are used to look up entries of an indexing and of the dimensions, hence must be convertible to
 * @c std::size_t. Levels are accessed by column (@c level[col]) and split into runs by offset, which requires
 * random access and a known size.
 *
 * Semantic precondition: all positions of a level refer to indices of the same dimension.
 */
template< typename T >
concept level = std::ranges::random_access_range< T > &&std::ranges::sized_range< T >
	&&std::convertible_to< std::ranges::range_reference_t< T >, std::size_t >;

/**
 * @brief A partition: a list of levels of equal length.
 *
 * The k-th entries of all levels form the k-th column, the unit that the symmetry permutes. The levels are
 * traversed repeatedly and in reverse (the last level is the most significant one), hence the range must be
 * multi-pass, bidirectional and common.
 *
 * Semantic precondition: all levels have the same size.
 */
template< typename T >
concept partition = std::ranges::forward_range< T > &&std::ranges::bidirectional_range< T >
	&&std::ranges::common_range< T > &&level< std::ranges::range_reference_t< T > >;

/**
 * @brief A list of partitions acting on disjoint tensor indices.
 *
 * Partitions are traversed in both directions (e.g. the last partition varies fastest when enumerating an
 * orbit), hence the range must be bidirectional and common.
 */
template< typename T >
concept partition_list = std::ranges::bidirectional_range< T > &&std::ranges::common_range< T >
	&&partition< std::ranges::range_reference_t< T > >;

/**
 * @brief A partition list that can be reordered in place.
 *
 * Partitions are permuted as a whole, the levels within a partition are permuted, and the index positions
 * stored in the levels are swapped, so every one of these must be writable.
 */
template< typename T >
concept sortable_partition_list = partition_list< T > &&std::ranges::random_access_range< T > &&std::permutable<
	std::ranges::iterator_t< T > > &&std::permutable< std::ranges::iterator_t< std::ranges::range_reference_t< T > > >
	&&std::ranges::output_range< std::ranges::range_reference_t< std::ranges::range_reference_t< T > >, std::size_t >;

/**
 * @brief The dimension (extent) of every tensor index, addressed by index position.
 */
template< typename T >
concept dimensions = std::ranges::random_access_range< T > &&std::ranges::sized_range< T >
	&&std::convertible_to< std::ranges::range_reference_t< T >, std::size_t >;

/**
 * @brief An indexing whose values are only compared and moved around.
 *
 * Suffices for everything that does not interpret index values numerically (canonicalisation, orbit
 * enumeration). Entries are addressed via @c operator[] by index position. The value type must be totally
 * ordered, as columns are ordered by it.
 */
template< typename T >
concept tensor_indexing = std::ranges::random_access_range< T > &&requires(T &indexing, std::size_t pos) {
	indexing[pos];
}
&&std::totally_ordered< std::ranges::range_value_t< T > > &&std::copyable< std::ranges::range_value_t< T > >;

/**
 * @brief A tensor_indexing whose entries can be swapped, as required to bring it into another order.
 */
template< typename T >
concept mutable_tensor_indexing = tensor_indexing< T > &&std::permutable< std::ranges::iterator_t< T > >;

/**
 * @brief A tensor_indexing of plain integers, one per tensor index, as consumed by rank().
 */
template< typename T >
concept integral_indexing = tensor_indexing< T > &&std::ranges::sized_range< T >
	&&std::convertible_to< std::ranges::range_reference_t< T >, std::size_t >;

/**
 * @brief An indexing of plain integers that can be filled with index values, as produced by unrank().
 */
template< typename T >
concept writable_integral_indexing = integral_indexing< T > &&std::ranges::output_range< T, std::size_t >;

/**
 * @brief Scratch storage for the effective (merged) index of one partition at a time.
 *
 * Must hold at least as many entries as the largest number of columns of any partition.
 */
template< typename T >
concept index_scratch = std::ranges::random_access_range< T > &&std::ranges::sized_range< T > &&
	std::same_as< std::ranges::range_value_t< T >, std::size_t > &&
		std::ranges::output_range< T, std::size_t > &&requires(T &scratch, std::size_t pos) {
	scratch[pos];
};

/**
 * @brief The admissible types for the per-partition sign argument: the all_symmetric_t sentinel, a single
 *        Symmetry applied to every partition, or a random-access range of per-partition Symmetry flags.
 *
 * Constraining the sign parameter to this concept lets it coexist unambiguously with the other range
 * parameters of @c rank / @c unrank (effective-index scratch buffers of @c std::size_t), so a sign array and
 * a scratch buffer are never confused during overload resolution.
 *
 * Semantic precondition: a range holds exactly one flag per partition.
 */
template< typename T >
concept partition_signs =
	std::same_as<
		std::remove_cvref_t< T >,
		all_symmetric_t > || std::convertible_to< T, Symmetry > || (std::ranges::random_access_range< T > && std::same_as< std::ranges::range_value_t< T >, Symmetry >);

/**
 * @brief A symmetry label, e.g. an irrep of an abelian point group.
 *
 * Labels are copied, compared for equality, sorted and used as map keys.
 */
template< typename T > concept label = std::regular< T > &&std::three_way_comparable< T >;

/**
 * @brief A binary operation combining two labels into a label.
 */
template< typename T, typename Label >
concept label_combiner = std::regular_invocable< T &, Label, Label >
	&&std::convertible_to< std::invoke_result_t< T &, Label, Label >, Label >;

/**
 * @brief The labelled dimensions of all tensor indices.
 *
 * One entry per tensor index, each a range of tuple-like @c (label, count) entries (e.g. a
 * @c std::map<Label, std::size_t>) giving the number of local values available for the label.
 */
template< typename T, typename Label >
concept labeled_dimensions = std::ranges::input_range< T > &&
	std::ranges::input_range< std::ranges::range_reference_t< T > > &&requires(
		std::ranges::range_reference_t< std::ranges::range_reference_t< T > > entry) {
	{ std::get< 0 >(entry) }
	->std::convertible_to< Label >;
	{ std::get< 1 >(entry) }
	->std::convertible_to< std::size_t >;
};

/**
 * @brief An indexing of Labeled values, one per tensor index, as consumed by SectorLayout.
 */
template< typename T, typename Label >
concept labeled_indexing =
	std::ranges::random_access_range< T > &&std::same_as< std::ranges::range_value_t< T >, Labeled< Label > >;

/**
 * @brief A labeled_indexing that can be filled with values, as produced by SectorLayout::unrank().
 */
template< typename T, typename Label >
concept writable_labeled_indexing = labeled_indexing< T, Label > &&std::ranges::output_range< T, Labeled< Label > >;

} // namespace tpack
