# Keeping the examples honest

The code blocks in the docs are not hand-maintained prose: the ones tagged with a `tpack-snippet` directive are extracted by
[`docs/extract_snippets.py`](extract_snippets.py), compiled against the current headers, and (unless marked otherwise) registered as CTest tests, so
their `assert`s actually run. This happens as part of the normal top-level build — `cmake --build build` compiles them and `ctest -R '^TPack'` runs
them as `TPack.DocSnippet.<name>` — which means an API change that breaks an example breaks CI.

To make a code block part of this harness, precede a fenced `cpp` block with a directive comment:

````text
<!-- tpack-snippet: name=my_example -->
```cpp
std::vector<std::size_t> dims = {3};
assert(tpack::num_orbits(dims, std::vector<std::vector<std::vector<std::size_t>>>{{{0}}}) == 3);
```
````

Untagged code blocks (like this very illustration) are ignored, so prose fragments and shell transcripts are free to appear without being compiled.
Supported directive keys:

- `name` — unique identifier for the generated source and test (defaults to `<doc-stem>_<n>`).
- `mode` — `body` (the default; the block is wrapped in a generated `main()` with the common headers already included), `program` (the block is a
  complete translation unit with its own includes and `main`), or `fragment` (wrapped but only compiled, never run).
- `run` — `true`/`false` to override whether the snippet runs as a test.
- `headers` — extra standard headers for wrapped snippets, e.g. `headers="algorithm set"`.

Because `body`-mode snippets are wrapped in a `main()` that already includes the TPack headers plus `<cassert>`, `<cstddef>` and `<vector>`, those
examples can focus on the API itself.

