# Usage

TPack is header-only: add `include/` to your include path (or link the `TPack::TPack` CMake target) and include the headers you need. Everything lives
in namespace `tpack`; helpers under `tpack::details` are implementation details and not part of the public API.

| Header | Public API |
| --- | --- |
| `<tpack/orbit.hpp>` | `num_orbits`, `is_canonical`, `next_orbit_representative` |
| `<tpack/rank.hpp>` | `rank`, `unrank` |
| `<tpack/partition.hpp>` | `sort_partition_col_major`, `sort_partition_row_major` |

## Describing a symmetry

Every function takes the same two descriptors (see [Core vocabulary](README.md#core-vocabulary)):

- `dims` — a random-access range giving the dimension of each index.
- `partitions` — a range of partitions. A partition is a range of levels; a level is a range of index positions. All levels of one partition must have
  the same length, and every index inside a single level must share the same dimension.

For a symmetric matrix `A(i, j) == A(j, i)` over a `5 x 5` grid, indices `0` and `1` form one partition with a single level containing both:

```cpp
std::vector<std::size_t> dims = {5, 5};
std::vector<std::vector<std::vector<std::size_t>>> partitions = {{{0, 1}}};
```

A level with more than one entry describes a *joint* symmetry: for `partitions = {{{0, 1}, {2, 3}}}` the columns `(0, 2)` and `(1, 3)` are permuted
together, i.e. swapping index 0 with 1 forces index 2 to swap with 3 — the symmetry of a pair of index pairs such as a two-electron integral
`(ab|cd)`.

## Getting started

The following complete program fills a packed symmetric matrix, expands it to a dense one via `unrank` + `next_orbit_representative`, and checks the
symmetry holds.

<!-- tpack-snippet: name=usage_getting_started mode=program -->
```cpp
#include <tpack/orbit.hpp>
#include <tpack/rank.hpp>

#include <cassert>
#include <cstddef>
#include <vector>

int main() {
	const std::vector<std::size_t> dims = {4, 4};
	const std::vector<std::vector<std::vector<std::size_t>>> partitions = {{{0, 1}}};

	const std::size_t count = tpack::num_orbits(dims, partitions);

	// One value per orbit in the packed (minimal) storage.
	std::vector<int> packed(count);
	for (std::size_t r = 0; r < count; ++r) {
		packed[r] = static_cast<int>(r) + 1;
	}

	// Expand into a dense 4x4 matrix: write every orbit member from its canonical
	// representative obtained via unrank().
	std::vector<int> dense(dims[0] * dims[1], 0);
	for (std::size_t r = 0; r < count; ++r) {
		auto idx = tpack::unrank(r, dims, partitions);
		do {
			dense[idx[0] * dims[1] + idx[1]] = packed[r];
		} while (tpack::next_orbit_representative(idx, partitions));
	}

	// Every entry is populated and the matrix is symmetric.
	for (std::size_t i = 0; i < dims[0]; ++i) {
		for (std::size_t j = 0; j < dims[1]; ++j) {
			assert(dense[i * dims[1] + j] != 0);
			assert(dense[i * dims[1] + j] == dense[j * dims[1] + i]);
		}
	}

	return 0;
}
```

## The public functions

### `num_orbits`

```
std::size_t num_orbits(dims, partitions);
```

Returns how many distinct values the symmetric tensor has — exactly the size the packed storage needs. **Internally** it multiplies, over all
partitions, the multiset coefficient `C(effective_dim + num_cols - 1, num_cols)`, where `effective_dim` is the product of the per-level dimensions and
`num_cols` is the number of columns. This is the counting result derived in [Theory](theory.md#the-combinatorial-idea-combinatorial-numbers).

<!-- tpack-snippet: name=usage_num_orbits -->
```cpp
std::vector<std::size_t> dims = {5, 5};
std::vector<std::vector<std::vector<std::size_t>>> partitions = {{{0, 1}}};

assert(tpack::num_orbits(dims, partitions) == 15);  // C(5 + 2 - 1, 2) = C(6, 2)
```

### `rank` / `unrank`

```
std::size_t rank(idx, dims, parts);                 // canonical indexing -> packed offset
Indexing    unrank(rank, dims, parts);              // packed offset -> canonical indexing
void        unrank(idx, rank, dims, parts);         // write into a caller-provided buffer
```

`rank` maps a **canonical** indexing to its packed storage offset in `[0, num_orbits(dims, parts))`; `unrank` is the exact inverse and always returns
a canonical indexing. **Internally** each partition is collapsed into a single effective 1-D symmetric index; `rank` evaluates the
combinatorial-number-system sum over its non-increasing columns, while `unrank` runs the greedy inverse, subtracting the largest binomial coefficient
not exceeding the remaining rank. Independent partitions are combined with a mixed-radix stride (the product of the preceding partitions' orbit
counts).

Together they form a bijection between orbits and `[0, count)`:

<!-- tpack-snippet: name=usage_rank_unrank -->
```cpp
std::vector<std::size_t> dims = {3, 3, 5, 5};
// One partition, two levels: the columns (0,2) and (1,3) are symmetric together.
std::vector<std::vector<std::vector<std::size_t>>> partitions = {{{0, 1}, {2, 3}}};

std::size_t count = tpack::num_orbits(dims, partitions);
for (std::size_t r = 0; r < count; ++r) {
	std::vector<std::size_t> idx = tpack::unrank(r, dims, partitions);

	assert(tpack::is_canonical(idx, partitions));
	assert(tpack::rank(idx, dims, partitions) == r);
}
```

Passing a non-canonical indexing to `rank` is a precondition violation (checked by assertions); canonicalise it first with
`next_orbit_representative`. Both functions also offer an overload taking an explicit scratch buffer (`effective_idx`) to avoid the internal
thread-local allocation in hot loops; those overloads are `constexpr`.

### `is_canonical` / `next_orbit_representative`

```
bool is_canonical(idx, partitions);
bool next_orbit_representative(idx, parts, counters = nullptr);
```

`is_canonical` reports whether an indexing is the chosen representative for its orbit TPack chose for storage — namely one whose columns are
non-increasing in reverse-lexicographic order within every partition.

`next_orbit_representative` walks the orbit: starting from any member, each call advances `idx` in place to the next member and returns `true`; the
call that exhausts the orbit returns `false` and restores `idx` to the canonical representative. It is the tool for _scattering_ a packed value to all
the dense positions it represents (as in the getting-started example) and for canonicalising an arbitrary indexing.

<!-- tpack-snippet: name=usage_enumerate -->
```cpp
std::vector<std::size_t> dims = {4, 4, 4};
std::vector<std::vector<std::vector<std::size_t>>> partitions = {{{0, 1, 2}}};

std::vector<std::size_t> idx = {2, 1, 0};        // canonical: non-increasing
assert(tpack::is_canonical(idx, partitions));

std::size_t members = 0;
do {
	++members;
} while (tpack::next_orbit_representative(idx, partitions));

assert(members == 6);                            // 3! distinct permutations
assert((idx == std::vector<std::size_t>{2, 1, 0}));  // restored to canonical
```

The optional `counters` argument (one counter per partition) is not needed for symmetric packing; see the API reference for its meaning.

### `sort_partition_col_major` / `sort_partition_row_major`

```
void sort_partition_col_major(partitions);
void sort_partition_row_major(partitions);
```

These sort the way a partition is expressed so that accessing the represented tensor (during read or write) happens in accordance with its storage
order (as far as possible, that is). As a side-effect, they normalise a partition specification in place so that specifications describing the same
symmetry become identical, which keeps the `rank`/`unrank` layout stable regardless of the order the caller happened to list indices, levels and
partitions in.

<!-- tpack-snippet: name=usage_sort_partition -->
```cpp
std::vector<std::vector<std::vector<std::size_t>>> a = {{{0, 1}}};
std::vector<std::vector<std::vector<std::size_t>>> b = {{{1, 0}}};

// Both describe the same symmetric pair; normalising makes them identical.
tpack::sort_partition_col_major(a);
tpack::sort_partition_col_major(b);
assert(a == b);
```

