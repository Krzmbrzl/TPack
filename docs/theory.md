# Theory

This page sketches the mathematical background behind TPack and then explains
the idea that the library is built on. It is deliberately a sketch: the
group-theory concepts are standard and are covered far more thoroughly by the
linked references.

## Group-theoretic background (sketched)

A tensor with permutation symmetry is a tensor whose value is unchanged when
certain of its indices are permuted. For example, a symmetric matrix satisfies
`A(i, j) == A(j, i)`, and a fully symmetric rank-3 tensor satisfies
`T(i, j, k) == T(j, i, k) == ...` for every permutation of `i, j, k`.

The permutations that leave the tensor invariant form a
[**group**](https://en.wikipedia.org/wiki/Group_(mathematics)) — a subgroup of
the [symmetric group](https://en.wikipedia.org/wiki/Symmetric_group) `S_n` on
the `n` indices involved. That group **acts** on the set of all indexings (see
[group action](https://en.wikipedia.org/wiki/Group_action)). Under a group
action, the index set splits into disjoint
[**orbits**](https://en.wikipedia.org/wiki/Group_action#Orbits_and_stabilizers):
two indexings lie in the same orbit exactly when some symmetry permutation maps
one to the other. Because the tensor value is constant along an orbit, storing
**one representative per orbit** is enough to recover every element.

Full symmetry makes the representative and counting problem purely
combinatorial:

- **Fully symmetric** — every permutation of the group's indices is a symmetry,
  and the value is unchanged. The acting group is all of `S_n`.

For anything less than total symmetry — so-called *mixed symmetry*, described by
[Young tableaux](https://en.wikipedia.org/wiki/Young_tableau) and the
[representation theory of the symmetric group](https://en.wikipedia.org/wiki/Representation_theory_of_the_symmetric_group)
— the orbit structure is not a simple counting problem, and the closed-form
addressing scheme below no longer applies. **This is why TPack only supports
fully symmetric partitions.** Each partition in the API is one independent fully
symmetric group acting on its own columns; different partitions act on disjoint
index sets, so their effects simply multiply.

## The combinatorial idea: combinatorial numbers

For a fully symmetric group `S_k` acting on `k` columns, the orbits have a
canonical representative that is trivial to describe: **sort the columns** into
non-increasing order.

Two indexings are in the same orbit iff they are permutations of each other, so
each orbit contains exactly one non-increasing ordering. Counting orbits is
therefore counting non-increasing length-`k` sequences over the `d` possible
column values — equivalently the number of
[multisets](https://en.wikipedia.org/wiki/Multiset) of size `k` drawn from `d`
symbols, i.e.
[combinations with repetition](https://en.wikipedia.org/wiki/Combination#Number_of_combinations_with_repetition):

```
C(d + k - 1, k)      ("stars and bars")
```

Here `d` is the **effective dimension** of a column: a column is a tuple with
one entry per level, so its combined range is the product of the per-level
dimensions. (This requires every index within a level to have the same
dimension — a precondition the `rank`/`unrank` functions assert.)

The key move is that once orbits are in bijection with sorted sequences, we can
address them with a [**combinatorial number system**](https://en.wikipedia.org/wiki/Combinatorial_number_system)
(also called *combinadics*). The combinatorial number system assigns to every
sorted sequence a unique integer — its **rank** — in `[0, N)` where `N` is the
number of sequences, using sums of binomial coefficients:

```
rank(x_0 >= x_1 >= ... >= x_{k-1}) = sum_i  C(x_i + (k - i) - 1, k - i)
```

Because this is a bijection, it inverts: given a rank, a greedy procedure
recovers the sorted sequence one column at a time by subtracting the largest
binomial coefficient that does not exceed the remaining rank. This is exactly
what TPack's [`rank`](usage.md#rank--unrank) and
[`unrank`](usage.md#rank--unrank) do per partition, combining the independent
partitions with a mixed-radix stride. No table of representatives is ever
materialised — the addressing is computed directly from the binomial
coefficients, which is what makes the scheme fast and allocation-light.

## Why this saves memory

Consider a rank-4 tensor with two independent symmetric index pairs, each index
ranging over dimension 10. Naively it has `10^4 = 10000` entries, but each
symmetric pair only has `C(10 + 2 - 1, 2) = C(11, 2) = 55` distinct values, so
the packed tensor needs `55 * 55 = 3025` — under a third of the dense size, and
the saving compounds with tensor rank and symmetry.

<!-- tpack-snippet: name=theory_packed_vs_dense -->
```cpp
// A rank-4 tensor with two symmetric pairs (0,1) and (2,3), each index of dim 10.
std::vector<std::size_t> dims = {10, 10, 10, 10};
std::vector<std::vector<std::vector<std::size_t>>> partitions = {{{0, 1}}, {{2, 3}}};

std::size_t dense  = 10 * 10 * 10 * 10;            // 10000
std::size_t packed = tpack::num_orbits(dims, partitions);

assert(packed == 55 * 55);                         // C(11,2)^2 = 3025
assert(packed < dense);
```

See the [Usage](usage.md) page for how to turn this counting result into actual
packed storage and back.
