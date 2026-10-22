# Usage

TPack is header-only: add `include/` to your include path (or link the `TPack::TPack` CMake target) and include the headers you need. Everything lives
in namespace `tpack`; helpers under `tpack::details` are implementation details and not part of the public API. Include `<tpack/tpack.hpp>` to pull in
the entire public API at once, or include the individual headers below.

| Header | Public API |
| --- | --- |
| `<tpack/tpack.hpp>` | umbrella header including all of the below |
| `<tpack/orbit.hpp>` | `num_orbits`, `orbit_size`, `next_orbit_representative`, `OrbitStep` |
| `<tpack/indexing.hpp>` | `is_canonical`, `canonicalize` |
| `<tpack/rank.hpp>` | `rank`, `unrank` |
| `<tpack/partition.hpp>` | `sort_partition`, `IndexOrder` |
| `<tpack/labeled.hpp>` | `Labeled` |
| `<tpack/sector_layout.hpp>` | `SectorLayout`, `Constraint` |

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
std::size_t num_orbits(dims, partitions, signs = {});
```

Returns how many distinct values the symmetric tensor has — exactly the size the packed storage needs. **Internally** it multiplies, over all
partitions, the multiset coefficient `C(effective_dim + num_cols - 1, num_cols)`, where `effective_dim` is the product of the per-level dimensions and
`num_cols` is the number of columns. This is the counting result derived in [Theory](theory.md#the-combinatorial-idea-combinatorial-numbers). The
optional `signs` argument selects antisymmetric partitions; see [Antisymmetric partitions](#antisymmetric-partitions).

<!-- tpack-snippet: name=usage_num_orbits -->
```cpp
std::vector<std::size_t> dims = {5, 5};
std::vector<std::vector<std::vector<std::size_t>>> partitions = {{{0, 1}}};

assert(tpack::num_orbits(dims, partitions) == 15);  // C(5 + 2 - 1, 2) = C(6, 2)
```

### `rank` / `unrank`

```
std::size_t rank(idx, dims, parts, signs = {});        // canonical indexing -> packed offset
Indexing    unrank(rank, dims, parts, signs = {});     // packed offset -> canonical indexing
void        unrank(rank, idx, dims, parts, signs = {});// write into a caller-provided buffer
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
`canonicalize`. Both functions also offer an overload taking an explicit scratch buffer (`effective_idx`) to avoid the internal
thread-local allocation in hot loops; those overloads are `constexpr`.

### `is_canonical` / `next_orbit_representative`

```
bool      is_canonical(idx, partitions, signs = {});
OrbitStep next_orbit_representative(idx, parts, signs = {}, phase = 1);
```

`is_canonical` reports whether an indexing is the chosen representative for its orbit TPack chose for storage — namely one whose columns are
non-increasing in reverse-lexicographic order within every partition (strictly decreasing for antisymmetric partitions).

`next_orbit_representative` walks the orbit: starting from any member, each call advances `idx` in place to the next member and returns an `OrbitStep`
that is `true` while further members remain; the call that exhausts the orbit returns an `OrbitStep` that is `false` and restores `idx` to the canonical
representative. The returned `OrbitStep::phase` carries the antisymmetric sign of the advanced `idx` (feed it back in as the next call's `phase`; it is
`+1` at the canonical representative and can be ignored for a purely symmetric orbit). It is the tool for _scattering_ a packed value to all the dense
positions it represents (as in the getting-started example). To turn an arbitrary indexing into its canonical form directly, use `canonicalize`.

<!-- tpack-snippet: name=usage_enumerate headers="iostream" -->
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
std::cout << "Number of members: " << members << "\n";
```

The optional `signs` and `phase` arguments track the antisymmetric sign while enumerating; they are not needed for symmetric packing. See
[Antisymmetric partitions](#antisymmetric-partitions).

### `orbit_size`

```
std::size_t orbit_size(idx, partitions, signs = {}, indexing_canonical = false);
```

`orbit_size` returns how many indexings share the orbit of `idx` — i.e. how many dense positions a single packed value expands to, which is exactly the
number of members `next_orbit_representative` would visit. It depends only on the column multiplicities, so it works from *any* member of the orbit and
leaves `idx` unchanged. **Internally** it multiplies, over all partitions, the multinomial coefficient `num_cols! / (m₁! · m₂! · …)`, where the `mₖ` are
the multiplicities of the distinct columns. Pass `indexing_canonical = true` when `idx` is already canonical to skip the internal sort.

<!-- tpack-snippet: name=usage_orbit_size -->
```cpp
std::vector<std::vector<std::vector<std::size_t>>> partitions = {{{0, 1, 2}}};

std::vector<std::size_t> idx = {2, 1, 0};             // three distinct columns
assert(tpack::orbit_size(idx, partitions) == 6);      // 3! members in this orbit

std::vector<std::size_t> repeated = {2, 1, 1};        // one repeated column
assert(tpack::orbit_size(repeated, partitions) == 3); // 3! / 2!
```

### `canonicalize`

```
int canonicalize(idx, partitions, signs = {});
```

`canonicalize` rewrites `idx` in place into the canonical representative of its orbit by sorting the columns of every partition into non-increasing
reverse-lexicographic order. After it returns, `is_canonical(idx, partitions)` holds, and any two members of the same orbit are mapped to the same
indexing. This is the direct way to put an arbitrary indexing into the form `rank` requires. The returned `int` is the antisymmetric phase relating the
input to the canonical representative (`+1` for symmetric packing); see [Antisymmetric partitions](#antisymmetric-partitions).

<!-- tpack-snippet: name=usage_canonicalize -->
```cpp
std::vector<std::size_t> dims = {4, 4, 4};
std::vector<std::vector<std::vector<std::size_t>>> partitions = {{{0, 1, 2}}};

std::vector<std::size_t> idx = {0, 2, 1};            // not canonical
assert(!tpack::is_canonical(idx, partitions));

tpack::canonicalize(idx, partitions);

assert(tpack::is_canonical(idx, partitions));
assert((idx == std::vector<std::size_t>{2, 1, 0}));  // columns now non-increasing

// rank() now accepts the canonicalised indexing.
assert(tpack::rank(idx, dims, partitions) < tpack::num_orbits(dims, partitions));
```

### `sort_partition`

```
template <IndexOrder Order = IndexOrder::ColumnMajor>
void sort_partition(partitions, signs = {});
```

This sorts the way a partition is expressed so that accessing the represented tensor (during read or write) happens in accordance with its storage
order (as far as possible, that is). The `Order` template argument selects the storage order: `IndexOrder::ColumnMajor` (the default) or
`IndexOrder::RowMajor`. As a side-effect, it normalises a partition specification in place so that specifications describing the same symmetry become
identical, which keeps the `rank`/`unrank` layout stable regardless of the order the caller happened to list indices, levels and partitions in. If a
`signs` array is passed it is reordered in lockstep with the partitions, so each partition keeps its antisymmetry flag.

<!-- tpack-snippet: name=usage_sort_partition -->
```cpp
std::vector<std::vector<std::vector<std::size_t>>> a = {{{0, 1}}};
std::vector<std::vector<std::vector<std::size_t>>> b = {{{1, 0}}};

// Both describe the same symmetric pair; normalising makes them identical.
tpack::sort_partition(a);
tpack::sort_partition(b);
assert(a == b);
```

## Antisymmetric partitions

By default every partition is *symmetric*: permuting its columns leaves the tensor value unchanged. A partition can instead be *antisymmetric*, where
permuting its columns multiplies the value by the sign (parity) of the permutation — the behaviour of a Slater determinant or a wedge product. Since a
partition is a whole symmetric group acting on its columns, antisymmetry is a property of the partition as a whole.

Antisymmetry is passed via the trailing `signs` argument that every function accepts: a range with one `tpack::Symmetry` per partition
(`Symmetry::Antisymm` or `Symmetry::Symm`). A single `Symmetry` may be passed instead to apply it to every partition. A range-valued `signs` must have
exactly one entry per partition. Omitting the argument means all partitions are symmetric, so existing code is unaffected. Two things change for an
antisymmetric partition:

- **Vanishing.** If two of its columns are equal, the element is forced to zero and is not stored. Its canonical representatives are therefore the
  *strictly* decreasing column tuples, and `num_orbits`/`rank`/`unrank` count and address them *without repetition* (`C(d, k)` instead of
  `C(d + k - 1, k)`). `orbit_size` reports `0` for a vanishing indexing.
- **Phase.** An arbitrary indexing equals `(-1)^p` times its canonical representative. `canonicalize` returns that phase — `+1`, `-1`, or `0` when the
  element vanishes — and `next_orbit_representative` threads a running `phase` through its argument and `OrbitStep::phase` result (start it at `+1` at
  the canonical representative). `rank` itself
  needs no phase: canonicalise first, keep the returned sign, then rank the canonical indexing.

<!-- tpack-snippet: name=usage_antisymmetric headers="iostream" -->
```cpp
std::vector<std::size_t> dims = {4, 4};
std::vector<std::vector<std::vector<std::size_t>>> partitions = {{{0, 1}}};
std::vector<tpack::Symmetry> signs = {tpack::Symmetry::Antisymm};  // the pair is antisymmetric

// Antisymmetric pairs store only the strictly-decreasing tuples: C(4, 2) = 6.
assert(tpack::num_orbits(dims, partitions, signs) == 6);

// Swapping the columns flips the sign; the canonical form is the descending order.
std::vector<std::size_t> idx = {1, 3};                   // ascending -> one transposition
int phase = tpack::canonicalize(idx, partitions, signs);
assert((idx == std::vector<std::size_t>{3, 1}));
assert(phase == -1);
std::cout << "phase relating {1, 3} to its canonical form: " << phase << '\n';

// Equal columns vanish.
std::vector<std::size_t> diag = {2, 2};
assert(tpack::canonicalize(diag, partitions, signs) == 0);
assert(tpack::orbit_size(std::vector<std::size_t>{2, 2}, partitions, signs) == 0);
```

## Labeled (sector-graded) packing

Tensors with abelian point-group symmetry store each index not as a flat integer but as a `tpack::Labeled<Label>` value: a symmetry label (e.g. an
irrep) together with a local index enumerating the states that share that label. A group *selection rule* means only label tuples whose group product
equals a target survive; the rest of the tensor is structurally zero. See the [Theory](theory.md#labelled-tensors-and-sectors) page for the
background.

`tpack::SectorLayout<Label>` packs exactly the surviving entries. It groups storage into *sectors* — one contiguous block per allowed, canonical label
assignment — and maps a canonical labelled indexing to and from a contiguous rank in `[0, size())`. Within a sector every index's label is fixed, so
what remains is the ordinary integer packing of the local indices, which it delegates to [`num_orbits`/`rank`/`unrank`](#the-public-functions).

It is constructed from per-slot `label -> count` dimensions (any range of `{label, count}` pairs; labels absent from a slot simply do not occur
there), the usual partitions and `signs`, and a `tpack::Constraint{combine, identity, target}` describing the group product.

<!-- tpack-snippet: name=usage_labeled mode=program -->
```cpp
#include <tpack/labeled.hpp>
#include <tpack/sector_layout.hpp>

#include <cassert>
#include <cstddef>
#include <iostream>
#include <map>
#include <vector>

int main() {
	using Label = int;

	// Two indices, each carrying labels {0, 1} with two local values per label.
	const std::vector<std::map<Label, std::size_t>> dims = {
		{{0, 2}, {1, 2}},
		{{0, 2}, {1, 2}},
	};
	// The two indices form one symmetric pair.
	const std::vector<std::vector<std::vector<std::size_t>>> partitions = {{{0, 1}}};

	// Abelian selection rule: the XOR of the two labels must be 0 (the totally symmetric irrep).
	const auto xor_combine = [](Label a, Label b) { return a ^ b; };
	const tpack::SectorLayout<Label> layout(dims, partitions,
											tpack::Constraint{xor_combine, /*identity=*/0, /*target=*/0});

	// Only the label-diagonal sectors (0, 0) and (1, 1) satisfy the rule; each is a symmetric pair over two
	// locals, C(3, 2) = 3 entries, so the packed size is 6.
	assert(layout.size() == 6);
	assert(layout.sectors().size() == 2);

	// The sectors tile the storage contiguously.
	std::size_t offset = 0;
	for (const auto &sector : layout.sectors()) {
		assert(sector.offset == offset);
		offset += sector.size;
	}
	assert(offset == layout.size());
	std::cout << "Final offset: " << offset << "\n";

	// rank and unrank are exact inverses over [0, size()), and every unranked indexing is canonical.
	for (std::size_t r = 0; r < layout.size(); ++r) {
		const std::vector<tpack::Labeled<Label>> idx = layout.unrank(r);
		assert(layout.is_canonical(idx));
		assert(layout.rank(idx) == r);
		std::cout << "Computed rank " << layout.rank(idx) << " == " << r << "\n";
	}

	return 0;
}
```

`sectors()` exposes each block's `{labels, offset, size}` for block-wise processing. The canonical form is *sector-major* (labels are the major sort
key, so each sector stays contiguous); for single-level partitions it coincides with [`is_canonical`](#is_canonical--next_orbit_representative), but
for multi-level partitions it differs, so test and canonicalise labelled indexings against `SectorLayout::is_canonical`. Antisymmetry acts within a
sector, on the locals of equal-label column runs: two columns sharing a label but differing in local index are distinct, and only equal `(label,
local)` columns vanish.

