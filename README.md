# TensorPacking

TPack is a small, header-only C++20 library for **packing and unpacking tensors that carry index-permutation symmetries**. A fully symmetric group of
tensor indices stores a lot of redundant entries; TPack maps the non-redundant entries onto a dense, contiguous range of integers so you can store the
tensor in the minimal amount of memory and still address every element.

See [the docs](docs/README.md) for more information.
