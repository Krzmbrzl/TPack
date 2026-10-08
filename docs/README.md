# TPack Documentation

TPack is a small, header-only C++20 library for **packing and unpacking tensors
that carry index-permutation symmetries**. A fully symmetric group of tensor
indices stores a lot of redundant entries; TPack maps the non-redundant entries
onto a dense, contiguous range of integers so you can store the tensor in the
minimal amount of memory and still address every element.

## Contents

- **[Theory](theory.md)** — the group-theoretic background (sketched, with
  links to authoritative sources) and the combinatorial idea that makes the
  library work, including why it is restricted to fully symmetric partitions.
- **[Usage](usage.md)** — the public API (`num_orbits`, `rank`/`unrank`,
  `is_canonical`/`next_orbit_representative`, `sort_partition_*`), what each
  function does, how it works internally, and worked, compilable examples.

Per-function and per-class reference documentation lives in the headers as
Doxygen comments under [`include/tpack/`](../include/tpack).

## Core vocabulary

These terms are used throughout the documentation and the API:

- **Index** — one slot of the tensor. Index `i` ranges over `[0, dims[i])`.
- **Indexing** — a concrete value for every index, i.e. one tensor element.
  Represented as a random-access range of `std::size_t` of length `size(dims)`.
- **Partition** — a group of indices that share a permutation symmetry,
  described as a list of **levels**. Each level is a list of index positions,
  and all levels of a partition have the same length. Reading one entry from
  each level at the same position gives a **column**; the symmetry permutes the
  columns of a partition among themselves.
- **Orbit** — the set of indexings reachable from one another by applying the
  symmetry. TPack stores one representative (the **canonical** one) per orbit.
- **Rank** — the dense integer in `[0, num_orbits(dims, parts))` assigned to a
  canonical indexing; `unrank` is its inverse.

## Building and testing the examples

The code examples in these docs are not just illustrative: the tagged ones are
extracted and compiled (and most are run as tests) on CI, so they stay in sync
with the API. See [Usage → Keeping the examples honest](usage.md#keeping-the-examples-honest)
for how the harness works.
