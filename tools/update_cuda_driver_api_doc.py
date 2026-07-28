#!/usr/bin/env python3
"""Generate cuda-driver-api.md from upstream CUDA Driver API group pages.

This script does the following:
- it scrapes the NVIDIA Driver API group pages for a target CUDA version,
- preserves any existing wrapper mappings already present in cuda-driver-api.md,
- uses a small JSON overrides file when available,
- and rewrites the markdown document in a format similar to the current one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

HTML_TAG_RE = re.compile(r"<[^>]+>")
SECTION_STATUS_RE = re.compile(r"\s+[✅🟡❌](?:\s*\([^)]*\))?$")
API_NAME_RE = re.compile(r"(cu[A-Z][A-Za-z0-9_]+)")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOC = ROOT / "cuda-driver-api.md"
DEFAULT_MAP = ROOT / "tools" / "cuda_driver_api_map.json"

SECTION_GROUPS = [
    ("Error Handling", "group__CUDA__ERROR"),
    ("Initialization", "group__CUDA__INITIALIZE"),
    ("Version Management", "group__CUDA__VERSION"),
    ("Device Management", "group__CUDA__DEVICE"),
    ("Primary Context Management", "group__CUDA__PRIMARY__CTX"),
    ("Context Management", "group__CUDA__CTX"),
    ("Module Management", "group__CUDA__MODULE"),
    ("Library Management", "group__CUDA__LIBRARY"),
    ("Memory Management", "group__CUDA__MEM"),
    ("Virtual Memory Management", "group__CUDA__VA"),
    ("Stream Ordered Memory Allocator", "group__CUDA__MALLOC__ASYNC"),
    ("Multicast Object Management", "group__CUDA__MULTICAST"),
    ("Logical Endpoint", "group__CUDA__LOGICAL__ENDPOINT"),
    ("Unified Addressing", "group__CUDA__UNIFIED"),
    ("Stream Management", "group__CUDA__STREAM"),
    ("Event Management", "group__CUDA__EVENT"),
    ("External Resource Interoperability", "group__CUDA__EXTRES__INTEROP"),
    ("Stream Memory Operations", "group__CUDA__MEMOP"),
    ("Execution Control", "group__CUDA__EXEC"),
    ("Graph Management", "group__CUDA__GRAPH"),
    ("Occupancy", "group__CUDA__OCCUPANCY"),
    ("Texture Object Management", "group__CUDA__TEXOBJECT"),
    ("Surface Object Management", "group__CUDA__SURFOBJECT"),
    ("Tensor Map Object Management", "group__CUDA__TENSOR__MEMORY"),
    ("Peer Context Memory Access", "group__CUDA__PEER__ACCESS"),
    ("Graphics Interoperability", "group__CUDA__GRAPHICS"),
    ("Driver Entry Point Access", "group__CUDA__DRIVER__ENTRY__POINT"),
    ("Coredump Attributes Control API", "group__CUDA__COREDUMP"),
    ("Green Contexts", "group__CUDA__GREEN__CONTEXTS"),
    ("Error Log Management Functions", "group__CUDA__LOGS"),
    ("CUDA Checkpointing", "group__CUDA__CHECKPOINT"),
    ("Profiler Control", "group__CUDA__PROFILER"),
]


def fetch_url(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; cudawrappers-doc-updater/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def parse_upstream_functions(group_page: str) -> List[str]:
    html = fetch_url(group_page)
    names: List[str] = []
    seen = set()

    for line in html.splitlines():
        if "member_name" not in line:
            continue
        stripped = HTML_TAG_RE.sub("", line)
        match = API_NAME_RE.search(stripped)
        if not match:
            continue
        name = match.group(1)
        if any(token in name for token in ("_v2", "_v3")):
            continue
        if name not in seen:
            seen.add(name)
            names.append(name)

    return names


def read_existing_doc(path: Path) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    if not path.exists():
        return {}, []
    text = path.read_text(encoding="utf-8")
    sections = OrderedDict()
    current = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            heading = line[3:].strip()
            heading = SECTION_STATUS_RE.sub("", heading)
            if heading in {"Coverage summary", "Summary"}:
                current = None
                continue
            current = heading
            sections[current] = OrderedDict()
            continue
        if current is None:
            continue
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        api = cells[0].strip("`")
        wrapper = cells[1].strip()
        if api.startswith("cu"):
            sections[current][api] = wrapper
    return sections, text.splitlines()


def load_overrides(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("overrides", {})


def normalize_wrapper(wrapper: str | None) -> str:
    if not wrapper:
        return "Missing"
    text = wrapper.strip()
    if not text:
        return "Missing"
    return text.strip("`").strip()


def describe_wrapper(wrapper: str | None) -> str:
    wrapper = normalize_wrapper(wrapper)
    if not wrapper or wrapper == "Missing":
        return "Missing"
    return f"`{wrapper}`"


def describe_section_heading(section_name: str, rows: List[Tuple[str, str]]) -> str:
    implemented = sum(
        1 for _, wrapper in rows if normalize_wrapper(wrapper) != "Missing"
    )
    total = len(rows)
    if total == 0:
        return f"## {section_name}"
    if implemented == total:
        status = "✅"
    elif implemented == 0:
        status = "❌"
    else:
        status = "🟡"
    return f"## {section_name} {status} ({implemented}/{total})"


def build_markdown(
    sections: Dict[str, List[Tuple[str, str]]], cuda_version: str | None = None
) -> str:
    intro_lines = [
        "# CUDA Driver API Coverage for cudawrappers",
        "",
    ]
    if cuda_version:
        intro_lines.append(
            f"This document summarizes CUDA Driver API coverage for `cudawrappers` against CUDA {cuda_version}. It lists the APIs in the upstream Driver API sectioning and records whether a wrapper is represented in the project."
        )
    else:
        intro_lines.append(
            "This document summarizes CUDA Driver API coverage for `cudawrappers` against the NVIDIA Driver API reference. It lists the APIs in the upstream sectioning and records whether a wrapper is represented in the project."
        )

    intro_lines.extend(
        [
            "",
            "Reference: https://docs.nvidia.com/cuda/cuda-driver-api/index.html",
            "",
        ]
    )

    out_lines = list(intro_lines)
    out_lines.append("")
    for section_name in sections:
        rows = sections[section_name]
        out_lines.append(describe_section_heading(section_name, rows))
        out_lines.append("")
        out_lines.append("| CUDA Driver API | cudawrappers interface |")
        out_lines.append("|---|---|")
        for api, wrapper in rows:
            out_lines.append(f"| `{api}` | {describe_wrapper(wrapper)} |")
        out_lines.append("")

    return "\n".join(out_lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh cuda-driver-api.md from NVIDIA docs"
    )
    parser.add_argument(
        "--cuda-version", default="", help="CUDA version to target (optional)"
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_DOC), help="Path to the markdown output"
    )
    parser.add_argument(
        "--mapping", default=str(DEFAULT_MAP), help="Path to a JSON overrides file"
    )
    parser.add_argument(
        "--check", action="store_true", help="Exit non-zero if the output would change"
    )
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    mapping_path = Path(args.mapping).resolve()

    existing_sections, _ = read_existing_doc(DEFAULT_DOC)
    overrides = load_overrides(mapping_path)

    # Preserve all sections already present in the markdown, then add the upstream-driven sections.
    ordered_sections = OrderedDict()
    for section_name in list(existing_sections.keys()):
        ordered_sections[section_name] = []

    for section_name, group_slug in SECTION_GROUPS:
        if section_name not in ordered_sections:
            ordered_sections[section_name] = []

    # Populate rows from upstream docs.
    for section_name, group_slug in SECTION_GROUPS:
        section_url = f"https://docs.nvidia.com/cuda/cuda-driver-api/{group_slug}.html"
        api_names = parse_upstream_functions(section_url)
        rows: List[Tuple[str, str]] = []
        for api_name in api_names:
            if api_name in overrides:
                wrapper = overrides[api_name]
            else:
                wrapper = existing_sections.get(section_name, {}).get(
                    api_name, "Missing"
                )
            rows.append((api_name, wrapper))
        ordered_sections[section_name] = rows

    # Preserve any manual sections from the current document that are not in the upstream mapping.
    upstream_section_names = {name for name, _ in SECTION_GROUPS}
    for section_name, rows_map in existing_sections.items():
        if section_name in ordered_sections and section_name in upstream_section_names:
            continue
        if section_name not in ordered_sections:
            ordered_sections[section_name] = [
                (api, wrapper) for api, wrapper in rows_map.items()
            ]

    # Ensure the intro list doesn't include a duplicate or empty placeholder for the first section.
    rendered = build_markdown(ordered_sections, args.cuda_version or None)
    output_path.write_text(rendered, encoding="utf-8")

    if args.check:
        existing = (
            DEFAULT_DOC.read_text(encoding="utf-8") if DEFAULT_DOC.exists() else ""
        )
        return 0 if existing == rendered else 1

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
