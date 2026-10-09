# TPack Documentation

## Contents

- **[Theory](theory.md)** — the group-theoretic background and the combinatorial idea that makes the library work, including why it is restricted to
  fully symmetric partitions.
- **[Usage](usage.md)** — the public API , what each function does, how it works internally, and worked, compilable examples.

Per-function and per-class reference documentation lives in the headers as Doxygen comments under [`include/tpack/`](../include/tpack).

## Core vocabulary

These terms are used throughout the documentation and the API:

- **Index**: one slot of the tensor. Index `i` ranges over `[0, dims[i])`.
- **Indexing**: a concrete value for every index, i.e. one tensor element. Effectively a range of numbers where the n-th number represents a specific
  value for the n-th index.
- **Partition**: a description of the permutational symmetries of indices. A set of indices is partitioned into disjoint subsets where every subset
  can have some set of permutational symmetries among them. Every partition consists of a list of **levels**, where a level contains positions of
  indices that can be freely permuted with each other. If multiple levels are contained in a single partition, this means that when permuting the
  `i-th and j-th index of the first level, we have to simultaneously do so for all other levels. That is, indices have to be permuted in pairs
  (tuples). A set of indices that must always be moved together is referred to as a **column**. This name derives from the idea of writing every level
  in a new line so that the k-th index in every level is exactly the k-th _column_ of the resulting matrix. All in all, that means that permutational
  symmetries really work on level columns rather than individual indices (the latter is just a special case if there is only a single level in a given
  partition).
- **Orbit**: the set of indexings reachable from one another by applying the symmetry. TPack stores one representative (the **canonical** one) per
  orbit.
- **Rank**: the integer in `[0, total_rank)` assigned to a canonical indexing, where total_rank is the total number of orbits. The mapping from a
  canonical indexing to the associated integer is done by the `rank(…)` function, whereas `unrank(…)` is inverse operation.

