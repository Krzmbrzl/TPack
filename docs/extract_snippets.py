# SPDX-License-Identifier: BSD-3-Clause
"""Extract tagged C++ code snippets from the Markdown docs into compilable sources.

Only fenced ``cpp`` blocks that are immediately preceded by a

    <!-- tpack-snippet: key=value ... -->

directive are extracted; every other code block (shell transcripts, pseudo-code,
untagged examples) is ignored. This keeps the docs free to show illustrative
fragments while guaranteeing that the tagged snippets keep compiling (and, where
requested, keep passing their assertions) against the current API on CI.

Supported directive keys:
  name     Identifier for the generated source and its test (defaults to
           "<doc-stem>_<n>"). Must be unique across all docs.
  mode     program  - the block is a complete translation unit (own includes/main)
           body     - the block is wrapped in a generated main() (the default)
           fragment - like body, but only compiled, never run
  run      true|false - override whether the snippet is executed as a test
           (defaults to true for program/body, false for fragment)
  headers  Space/comma separated extra standard headers to include in wrapped
           snippets, e.g. headers="algorithm,set"

The script writes one <name>.cpp per snippet into --out-dir and a CMake manifest
(--manifest) describing the snippets and whether each should be run.
"""

import argparse
import re
import shlex
import sys
from pathlib import Path

FENCE_RE = re.compile(r"^(?P<indent>\s*)(?P<fence>`{3,}|~{3,})\s*(?P<lang>[^\s`]*)\s*$")
DIRECTIVE_RE = re.compile(r"^\s*<!--\s*tpack-snippet:(?P<args>.*?)-->\s*$")

CPP_LANGS = {"cpp", "c++", "cc", "cxx"}

# Headers made available to every wrapped (body/fragment) snippet so that the
# docs do not need to repeat the boilerplate includes.
DEFAULT_WRAP_HEADERS = [
    "tpack/orbit.hpp",
    "tpack/partition.hpp",
    "tpack/rank.hpp",
    "cassert",
    "cstddef",
    "vector",
]


class SnippetError(RuntimeError):
    pass


def parse_directive(args_text):
    options = {}
    for token in shlex.split(args_text):
        if "=" not in token:
            raise SnippetError(f"malformed directive token '{token}' (expected key=value)")
        key, value = token.split("=", 1)
        options[key.strip()] = value.strip()
    return options


def iter_snippets(doc_path):
    """Yield (options, body_lines, body_start_line) for every tagged cpp block."""
    lines = doc_path.read_text(encoding="utf-8").splitlines()

    pending = None  # directive options waiting to attach to the next fence
    i = 0
    while i < len(lines):
        line = lines[i]

        directive = DIRECTIVE_RE.match(line)
        if directive:
            pending = parse_directive(directive.group("args"))
            i += 1
            continue

        fence = FENCE_RE.match(line)
        if not fence or not fence.group("lang"):
            # A blank line between directive and fence is fine; any other
            # non-fence content detaches a dangling directive.
            if pending is not None and line.strip():
                pending = None
            i += 1
            continue

        # Found the opening fence of a code block; locate its closing fence. A
        # closing fence uses the same character, carries no info string, and is
        # at least as long as the opener (CommonMark rules). The length check
        # lets the docs show nested fences by wrapping them in a longer fence.
        open_fence = fence.group("fence")
        body_start = i + 1
        j = body_start
        while j < len(lines):
            end = FENCE_RE.match(lines[j])
            if (
                end
                and not end.group("lang")
                and end.group("fence")[0] == open_fence[0]
                and len(end.group("fence")) >= len(open_fence)
            ):
                break
            j += 1
        if j >= len(lines):
            raise SnippetError(f"{doc_path.name}: unterminated code fence opened on line {i + 1}")

        if pending is not None:
            lang = fence.group("lang").lower()
            if lang not in CPP_LANGS:
                raise SnippetError(
                    f"{doc_path.name}: tpack-snippet directive on non-cpp block (lang '{lang}') on line {i + 1}"
                )
            yield pending, lines[body_start:j], body_start + 1
            pending = None

        i = j + 1

    if pending is not None:
        raise SnippetError(f"{doc_path.name}: tpack-snippet directive not followed by a code block")


def render_source(options, body_lines, doc_name, body_start_line):
    mode = options.get("mode", "body")
    if mode not in {"program", "body", "fragment"}:
        raise SnippetError(f"unknown mode '{mode}'")

    header = f"// Auto-generated from docs/{doc_name} -- do not edit.\n"
    line_marker = f'#line {body_start_line} "{doc_name}"\n'
    body = "\n".join(body_lines)

    if mode == "program":
        return header + line_marker + body + "\n"

    includes = list(DEFAULT_WRAP_HEADERS)
    extra = options.get("headers", "")
    for name in re.split(r"[,\s]+", extra):
        if name:
            includes.append(name)

    include_block = "".join(f"#include <{name}>\n" for name in includes)
    return (
        header
        + include_block
        + "\n"
        + "int main() {\n"
        + line_marker
        + body
        + "\n\treturn 0;\n}\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    snippets = []  # (name, run)
    seen_names = {}

    for doc_path in sorted(args.docs_dir.glob("*.md")):
        counter = 0
        for options, body_lines, body_start_line in iter_snippets(doc_path):
            counter += 1
            name = options.get("name") or f"{doc_path.stem}_{counter}"
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise SnippetError(f"{doc_path.name}: invalid snippet name '{name}'")
            if name in seen_names:
                raise SnippetError(
                    f"duplicate snippet name '{name}' in {doc_path.name} and {seen_names[name]}"
                )
            seen_names[name] = doc_path.name

            mode = options.get("mode", "body")
            run_default = mode != "fragment"
            run = options.get("run", str(run_default)).lower() in {"true", "1", "yes", "on"}

            source = render_source(options, body_lines, doc_path.name, body_start_line)
            (args.out_dir / f"{name}.cpp").write_text(source, encoding="utf-8")
            snippets.append((name, run))

    def cmake_bool(value):
        return "TRUE" if value else "FALSE"

    manifest_lines = [
        "# Auto-generated by extract_snippets.py -- do not edit.",
        'set(TPACK_DOC_SNIPPETS "{}")'.format(";".join(name for name, _ in snippets)),
    ]
    for name, run in snippets:
        manifest_lines.append(f"set(TPACK_DOC_SNIPPET_{name}_RUN {cmake_bool(run)})")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    print(f"extract_snippets: wrote {len(snippets)} snippet(s) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SnippetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
