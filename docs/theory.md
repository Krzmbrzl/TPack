# Theory

This page sketches the mathematical background behind TPack and then explains the idea that the library is built on. It is deliberately a sketch: the
group-theory concepts are standard and are covered far more thoroughly by the linked references.

## Group-theoretic background

A tensor with permutation symmetry is a tensor whose value is unchanged when certain of its indices are permuted. For example, a symmetric matrix
satisfies `A(i, j) == A(j, i)`, and a fully symmetric rank-3 tensor satisfies `T(i, j, k) == T(j, i, k) == ...` for every permutation of `i, j, k`.

The permutations that leave the tensor invariant form a [**group**](https://en.wikipedia.org/wiki/Group_(mathematics)) — a subgroup of the
[symmetric group](https://en.wikipedia.org/wiki/Symmetric_group) `S_n` on the `n` indices involved. That group **acts** on the set of all indexings
(see [group action](https://en.wikipedia.org/wiki/Group_action)). Under a group action, the index set splits into disjoint
[**orbits**](https://en.wikipedia.org/wiki/Group_action#Orbits_and_stabilizers): two indexings lie in the same orbit exactly when some symmetry
permutation maps one to the other. Because the tensor value is constant along an orbit, storing **one representative per orbit** is enough to recover
every element.

Full permutational symmetry makes the representative and counting problem purely combinatorial:

- **Fully symmetric** — every permutation of the group's indices is a symmetry, and the value is unchanged. The acting group is all of `S_n`.
- **Fully antisymmetric** — every permutation is again a symmetry of the *magnitude*, but the value picks up the permutation's sign. The acting group
  is still all of `S_n`, now acting with a sign (see [Antisymmetry](#antisymmetry)).

For anything less than total (anti)symmetry — so-called *mixed symmetry*, described by [Young tableaux](https://en.wikipedia.org/wiki/Young_tableau)
and the [representation theory of the symmetric group](https://en.wikipedia.org/wiki/Representation_theory_of_the_symmetric_group) — the orbit
structure is not a simple counting problem, and the closed-form addressing scheme below no longer applies. **This is why TPack only supports fully
symmetric or antisymmetric partitions.** Each partition in the API is one independent fully symmetric group acting on its own columns; different
partitions act on disjoint index sets, so their effects simply multiply.

## The combinatorial idea: combinatorial numbers

For a fully symmetric group `S_k` acting on `k` columns, the orbits have a canonical representative that is trivial to describe: **sort the columns**
into non-increasing order.

Two indexings are in the same orbit iff they are permutations of each other, so each orbit contains exactly one non-increasing ordering. Counting
orbits is therefore counting non-increasing length-`k` sequences over the `d` possible column values — equivalently the number of
[multisets](https://en.wikipedia.org/wiki/Multiset) of size `k` drawn from `d` symbols, i.e
[combinations with repetition](https://en.wikipedia.org/wiki/Combination#Number_of_combinations_with_repetition):

```
C(d + k - 1, k)      ("stars and bars")
```

Here `d` is the **effective dimension** of a column: a column is a tuple with one entry per level, so its combined range is the product of the
per-level dimensions. (This requires every index within a level to have the same dimension — a precondition the `rank`/`unrank` functions assert.)

The key move is that once orbits are in bijection with sorted sequences, we can address them with a
[**combinatorial number system**](https://en.wikipedia.org/wiki/Combinatorial_number_system) (also called *combinadics*). The combinatorial number
system assigns to every sorted sequence a unique integer — its **rank** — in `[0, N)` where `N` is the number of sequences/orbits, using sums of
binomial coefficients:

```
rank(x_0 >= x_1 >= ... >= x_{k-1}) = sum_i  C(x_i + (k - i) - 1, k - i)
```

Because this is a bijection, it inverts: given a rank, a greedy procedure recovers the sorted sequence one column at a time by subtracting the largest
binomial coefficient that does not exceed the remaining rank. This is exactly what TPack's [`rank`](usage.md#rank--unrank) and
[`unrank`](usage.md#rank--unrank) do per partition, combining the independent partitions with a mixed-radix stride. No table of representatives is
ever materialised — the addressing is computed directly from the binomial coefficients, which is what makes the scheme fast and allocation-light.

## Antisymmetry

A partition can instead be **antisymmetric**: permuting its columns multiplies the tensor value by the *sign* of the permutation. The acting group is
still the full symmetric group `S_k`, but it now acts on the value with a sign, as for a [Slater determinant](https://en.wikipedia.org/wiki/Slater_determinant)
or an [exterior (wedge) product](https://en.wikipedia.org/wiki/Exterior_algebra). Two things follow:

- Any permutation that fixes the indexing (a repeated column gives a transposition that does) is an *odd* symmetry of a would-be non-zero value, which
  is only consistent with the value being **zero**. So indexings with a repeated column vanish and are not stored.
- The surviving representatives are the **strictly decreasing** column tuples. Counting and addressing them is the same combinatorial-number system,
  only *without repetition*: the orbit count per partition becomes `C(d, k)` and the rank is `sum_i C(x_i, k - i)`.

Everything else is unchanged; a per-partition flag selects antisymmetric partitions, and the only extra output is the sign (phase) relating an
arbitrary indexing to its canonical representative. See the [Usage](usage.md#antisymmetric-partitions) page.

## Why this saves memory

Consider a rank-4 tensor with two independent symmetric index pairs, each index ranging over dimension 10. Naively it has `10^4 = 10000` entries, but
each symmetric pair only has `C(10 + 2 - 1, 2) = C(11, 2) = 55` distinct values, so the packed tensor needs `55 * 55 = 3025` — under a third of the
dense size, and the saving compounds with tensor rank and symmetry.

<!-- tpack-snippet: name=theory_packed_vs_dense headers="iostream" -->
```cpp
// A rank-4 tensor with two symmetric pairs (0,1) and (2,3), each index of dim 10.
std::vector<std::size_t> dims = {10, 10, 10, 10};
std::vector<std::vector<std::vector<std::size_t>>> partitions = {{{0, 1}}, {{2, 3}}};

std::size_t dense  = 10 * 10 * 10 * 10;            // 10000
std::size_t packed = tpack::num_orbits(dims, partitions);

assert(packed == 55 * 55);                         // C(11,2)^2 = 3025
assert(packed < dense);

std::cout << "dense: " << dense << ", packed: " << packed << '\n';
```

See the [Usage](usage.md) page for how to turn this counting result into actual packed storage and back.

## Labelled tensors and sectors

Tensors with abelian point-group symmetry carry, on every index, an irrep *label* in addition to a *local* index
within that label; TPack represents the pair as `Labeled<Label>`. Two things change relative to the flat case.

First, a group *selection rule*: the group product — for a point group, the XOR of the irrep bitmasks — of all the
indices' labels must equal a fixed target, or the entry is structurally zero. Only a sparse subset of label tuples
carries any data.

Second, the symmetry permutes the whole `(label, local)` pairs. The canonical representatives therefore fall into
**sectors**: one per label assignment that satisfies the rule and is canonical under the ordinary (symmetric)
column ordering. Storage is the concatenation of the sectors, each a contiguous block — exactly what
[`SectorLayout`](usage.md#labeled-sector-graded-packing) precomputes.

Within a sector every label is fixed, so the problem collapses back to the flat one. The local index of each slot
ranges over a plain integer dimension, and each partition splits into runs of equal-label columns — the
[Young subgroup](https://en.wikipedia.org/wiki/Young_subgroup) of column permutations that preserve the labels.
Each run is an ordinary (anti)symmetric partition over the locals, counted and addressed by exactly the
combinatorial-number machinery above. A sector's size is thus the product, over the partitions and their
equal-label runs, of the per-partition counts, and the layout's total size is the sum over the allowed sectors. No
new combinatorics are introduced; only the outer "which sector" layer is.

Because the whole point of the grading is that each sector is a contiguous block, the canonical ordering is
**sector-major** — labels first, then locals — rather than the per-level interleaving of label and local that the
flat `is_canonical` uses. The two agree exactly when every partition has a single level. Antisymmetry still acts
only through the local runs: two columns that share a label but differ in local index are distinct, and only equal
`(label, local)` columns vanish.

As in the flat case, this closed-form addressing covers only fully symmetric or antisymmetric partitions; mixed
(Young-tableau) symmetry remains out of scope.
