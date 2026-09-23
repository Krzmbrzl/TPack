# Notes for AI coding agents

Conventions for automated contributors working in this repository. Human
contributors: see `README.md`, which these notes do not replace.

## Keep commits to one logical change

When a change is implemented as a sequence of commits, give each commit
exactly one logical change — even within the same task, and even when a
later change builds directly on an earlier one. A logical change may touch
multiple files, but distinct bug fixes go in separate commits from each
other, a refactor that enables a later change is its own commit separate
from that change, and a new feature is its own commit separate from the
fix/refactor it depends on. Do not fold "fix A", "fix B", and "add feature C
that uses the fixed code" into one commit just because they happened in the
same session.

## Prefer the smallest diff that achieves the change

Make the smallest change that gets the job done, without compromising on
correctness or code readability. Don't fold in unrelated refactoring, cleanup,
or new abstractions the change doesn't strictly require, even if the
surrounding code looks like it could use it while you're in the area
— propose that separately and let it be its own change.

## Comments explain the present code; commit messages explain the change

Keep comments brief, and only write one where the *why* isn't obvious from
the code — a hidden constraint, a subtle invariant, a workaround. Don't use
a comment to narrate what the code used to do before this change, and don't
use a comment to justify why this change was made or why this approach was
chosen over an alternative. That reasoning belongs in the commit message.

## Reuse logic across similar call sites instead of duplicating it

Before adding a second implementation of behavior that already exists
elsewhere in the tree — even in a different header than the one being
added to — factor the shared part out into a function or utility both
call, rather than copying it. Implementation helpers that are not part of
the public API go under `include/tpack/details/`.

## Headers must be self-contained

TPack is header-only (`include/tpack/`), so a header missing an `#include`
can compile only because a test or benchmark TU happened to include the
missing header first. Every header includes what it uses. When adding a new
header, also add it to the `FILE_SET` in the top-level `CMakeLists.txt`,
otherwise it is not installed.

## Compiler Warnings

Top-level builds compile with `-Wall -Wextra -Wpedantic -Werror`, so a warning
is a build failure. Fix warnings rather than suppressing them, unless they are
clearly false-positives and fixing them would cause unreasonable code bloat or
performance/readability issues. Any non-trivially fixable warning must be
reported for further investigation by a human instead of being silently
ignored.

## Assertions

Assertions inside `assert()` are only conditionally enabled (disabled under
`NDEBUG`). Hence, code in assertions must not be relied on to be executed.
This may take some care with regards to unused-variable warnings (which are
errors here) when a variable is only used inside an assertion. In such cases
explicitly marking a variable as unused via a `(void)` cast or a
`[[maybe_unused]]` attribute solves the problem. Since CI builds without
`NDEBUG`, also check that code still compiles warning-free in a `Release`
configuration when touching assertions.

## Tests

Tests live under `tests/` (GoogleTest, fetched via `FetchContent` unless found
on the system) and build into `tensor_packing_tests`. Run them the way CI does
(`.github/workflows/build.yml`):

```
cmake -G Ninja -B build -S .
cmake --build build
cd build && ctest -R '^TPack' --output-on-failure
```

The `-R '^TPack'` filter excludes tests of dependencies; name new test
suites/instantiations so that they are matched by it.

Benchmarks (`benchmarks/`, Google Benchmark) build into
`tensor_packing_benchmarks` and are enabled by default for top-level builds;
they are not run by CI, but must keep compiling.

## Formatting

Formatting is defined by `.clang-format` (tabs for indentation, 120 column
limit). Format only the files you touched, never the whole tree:

```
clang-format --dry-run --Werror <files>   # check
clang-format -i <files>                   # fix
```

Different clang-format major versions can disagree; if running it produces
changes to lines you did not touch, revert those and only keep the formatting
of your own changes.
