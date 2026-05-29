#!/usr/bin/env python3
"""Export the FARBench dashboard as a GitHub Pages friendly static snapshot.

The live dashboard is FastAPI-backed and reads experiment records from disk.
GitHub Pages cannot run that backend, so this script precomputes the JSON
payloads used by the paper-facing dashboard views and rewrites dashboard.html
to read those local JSON files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENTS_DIR = Path("/data2/rab/RABench/experiments")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs"


TOP_LEVEL_ENDPOINTS = {
    "/api/agent-demo": "agent-demo.json",
    "/api/paper-analysis": "paper-analysis.json",
    "/api/analysis-cases": "analysis-cases.json",
    "/api/benchmark-tasks": "benchmark-tasks.json",
    "/api/experiments": "experiments.json",
    "/api/leaderboard": "leaderboard.json",
    "/api/capabilities": "capabilities.json",
    "/api/research-questions": "research-questions.json",
}

MAX_TEXT_CHARS = 24_000
MAX_DIFF_CHARS = 60_000
MAX_CODE_CHARS = 120_000
MAX_FILES_PER_ITERATION = 40
MAX_CURVE_POINTS = 5_000

# Binary / output-artifact files (model outputs, logs, checkpoints, media).
# Their content carries no value on the static dashboard and, once JSON-escaped,
# inflates the snapshot ~6x on disk. We keep the file's diff metadata but drop
# the embedded bytes, and we never export their code/ contents.
BINARY_EXTENSIONS = frozenset({
    # audio
    ".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac",
    # image
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".ico",
    # video
    ".mp4", ".avi", ".mov", ".mkv", ".webm",
    # arrays / model weights / serialized blobs
    ".npy", ".npz", ".pt", ".pth", ".ckpt", ".safetensors", ".bin",
    ".h5", ".hdf5", ".pkl", ".pickle", ".joblib",
    # archives
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar",
    # columnar data / misc binary
    ".parquet", ".arrow", ".feather", ".pdf",
})

# Substrings that mark TensorBoard event files (no stable extension).
BINARY_NAME_MARKERS = (".tfevents", "events.out.tfevents")


def _is_binary_artifact(path: Any) -> bool:
    if not isinstance(path, str) or not path:
        return False
    lowered = path.lower()
    if any(marker in lowered for marker in BINARY_NAME_MARKERS):
        return True
    return os.path.splitext(lowered)[1] in BINARY_EXTENSIONS


def _json_ready(value: Any) -> Any:
    """Convert values to strict JSON: no NaN/Inf, no sets/tuples."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_ready(v) for v in value)
    return value


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        # Minified: no indentation/whitespace. The dashboard parses JSON either
        # way; pretty-printing only inflated the snapshot (~13% across 30k files).
        json.dump(_json_ready(data), f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _truncate_text(value: Any, limit: int) -> tuple[Any, bool, int]:
    if not isinstance(value, str):
        return value, False, 0
    original_len = len(value)
    if original_len <= limit:
        return value, False, original_len

    marker = (
        f"\n\n[... static export truncated {original_len - limit} characters "
        f"from the middle; original length {original_len} characters ...]\n\n"
    )
    remaining = max(limit - len(marker), 0)
    head_len = remaining // 2
    tail_len = remaining - head_len
    return value[:head_len] + marker + value[-tail_len:], True, original_len


def _truncate_nested_strings(value: Any, limit: int) -> tuple[Any, bool, int]:
    if isinstance(value, str):
        return _truncate_text(value, limit)
    if isinstance(value, list):
        changed = False
        original = 0
        rows = []
        for item in value:
            slim, item_changed, item_original = _truncate_nested_strings(item, limit)
            rows.append(slim)
            changed = changed or item_changed
            original += item_original
        return rows, changed, original
    if isinstance(value, dict):
        changed = False
        original = 0
        out = {}
        for key, item in value.items():
            slim, item_changed, item_original = _truncate_nested_strings(item, limit)
            out[key] = slim
            changed = changed or item_changed
            original += item_original
        return out, changed, original
    return value, False, 0


def _slim_action(action: Any) -> Any:
    if not isinstance(action, dict):
        return action

    keep_keys = (
        "reasoning",
        "command",
        "submit_eval",
        "done",
        "done_reason",
        "checkpoint_path",
    )
    slim: dict[str, Any] = {}
    for key in keep_keys:
        if key in action:
            value, _, _ = _truncate_nested_strings(action[key], MAX_TEXT_CHARS)
            slim[key] = value

    files_to_write = action.get("files_to_write")
    if isinstance(files_to_write, dict):
        slim["files_written"] = sorted(str(path) for path in files_to_write)

    return slim


def _slim_command_output(command_output: Any) -> Any:
    if not isinstance(command_output, dict):
        return command_output

    slim = {key: value for key, value in command_output.items() if key not in {"stdout", "stderr"}}
    original_size = 0
    truncated = bool(command_output.get("_truncated"))
    for key in ("stdout", "stderr"):
        value, changed, original_len = _truncate_text(command_output.get(key), MAX_TEXT_CHARS)
        slim[key] = value
        original_size += original_len
        truncated = truncated or changed

    if truncated:
        slim["_truncated"] = True
        slim["_original_size"] = command_output.get("_original_size") or original_size
    return slim


def _slim_eval_result(eval_result: Any) -> Any:
    slim, changed, original_size = _truncate_nested_strings(eval_result, MAX_TEXT_CHARS)
    if changed and isinstance(slim, dict):
        slim["_truncated"] = True
        slim["_original_size"] = original_size
    return slim


def _slim_diff(diff_data: Any) -> Any:
    if not isinstance(diff_data, dict):
        return diff_data

    diffs = diff_data.get("diffs")
    if not isinstance(diffs, list):
        return diff_data

    slim_diffs = []
    omitted = 0
    truncated = False
    binary_stripped = 0
    for item in diffs[:MAX_FILES_PER_ITERATION]:
        if not isinstance(item, dict):
            continue
        slim = dict(item)
        if _is_binary_artifact(slim.get("filename")):
            slim["diff"] = "[binary file: content omitted from static export]"
            slim["_binary"] = True
            binary_stripped += 1
            slim_diffs.append(slim)
            continue
        diff_text, changed, original_len = _truncate_text(slim.get("diff"), MAX_DIFF_CHARS)
        slim["diff"] = diff_text
        if changed:
            slim["_truncated"] = True
            slim["_original_size"] = original_len
            truncated = True
        slim_diffs.append(slim)
    if len(diffs) > MAX_FILES_PER_ITERATION:
        omitted = len(diffs) - MAX_FILES_PER_ITERATION

    out = dict(diff_data)
    out["diffs"] = slim_diffs
    if omitted:
        out["_omitted_files"] = omitted
    if binary_stripped:
        out["_binary_files_stripped"] = binary_stripped
    if truncated:
        out["_truncated"] = True
    return out


def _sample_rows(rows: list[Any], limit: int) -> list[Any]:
    if len(rows) <= limit:
        return rows
    if limit <= 1:
        return rows[:limit]
    last = len(rows) - 1
    indexes = sorted({round(i * last / (limit - 1)) for i in range(limit)})
    return [rows[i] for i in indexes]


def _slim_curves(curve_data: Any) -> Any:
    if not isinstance(curve_data, dict):
        return curve_data
    rows = curve_data.get("data")
    if not isinstance(rows, list) or len(rows) <= MAX_CURVE_POINTS:
        return curve_data
    out = dict(curve_data)
    out["data"] = _sample_rows(rows, MAX_CURVE_POINTS)
    out["_truncated"] = True
    out["_original_points"] = len(rows)
    return out


def _candidate_files(iter_detail: Any, trajectory_row: Any, diff_data: Any) -> list[str]:
    candidates: list[str] = []

    def add_many(paths: Any) -> None:
        if isinstance(paths, dict):
            values = paths.keys()
        elif isinstance(paths, (list, tuple, set)):
            values = paths
        else:
            return
        for path in values:
            if isinstance(path, str) and path and path not in candidates:
                candidates.append(path)

    if isinstance(iter_detail, dict):
        action = iter_detail.get("action")
        if isinstance(action, dict):
            add_many(action.get("files_to_write"))
            add_many(action.get("files_written"))
    if isinstance(trajectory_row, dict):
        add_many(trajectory_row.get("files_written"))
    if isinstance(diff_data, dict):
        for item in diff_data.get("diffs") or []:
            if isinstance(item, dict):
                add_many([item.get("filename")])

    return candidates


def _slim_code_file(code_data: Any) -> Any:
    if not isinstance(code_data, dict):
        return code_data
    out = dict(code_data)
    content, changed, original_len = _truncate_text(out.get("content"), MAX_CODE_CHARS)
    out["content"] = content
    if changed:
        out["_truncated"] = True
        out["_original_size"] = original_len
    return out


def _slim_iteration_detail(
    iter_detail: Any,
    file_sources: dict[str, int],
    omitted_files: int,
) -> Any:
    if not isinstance(iter_detail, dict):
        return iter_detail
    # workspace_files is the cumulative set of files that exist as of this
    # iteration (carry-forward). file_sources maps each path to the iteration
    # whose code/ snapshot holds its current content — so an iteration that
    # changed nothing still shows the workspace inherited from earlier steps.
    return {
        "iteration": iter_detail.get("iteration"),
        "action": _slim_action(iter_detail.get("action")),
        "command_output": _slim_command_output(iter_detail.get("command_output")),
        "eval_result": _slim_eval_result(iter_detail.get("eval_result")),
        "observation": None,
        "workspace_files": sorted(file_sources),
        "file_sources": dict(file_sources),
        "agent_done": iter_detail.get("agent_done", False),
        "done_reason": iter_detail.get("done_reason"),
        "static_export": {
            "workspace_scope": "carry_forward_source_files",
            "omitted_files": omitted_files,
        },
    }


def _safe_call(endpoint: Any, **kwargs: Any) -> Any:
    try:
        return endpoint(**kwargs)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def _iteration_number(row: Any) -> int | None:
    if not isinstance(row, dict):
        return None
    try:
        return int(row.get("iteration"))
    except (TypeError, ValueError):
        return None


def _patch_dashboard_html(html: str) -> str:
    """Rewrite dashboard HTML so it can run from static files."""
    inject = (
        "<script>\n"
        "window.FARBENCH_STATIC_DASHBOARD = true;\n"
        "</script>\n"
    )
    html = html.replace("</head>", inject + "</head>", 1)

    replacements = {
        "fetch('/api/experiments')": "fetch('./api/experiments.json')",
        "fetch('/api/agent-demo',": "fetch('./api/agent-demo.json',",
        "fetch('/api/leaderboard',": "fetch('./api/leaderboard.json',",
        "fetch('/api/capabilities',": "fetch('./api/capabilities.json',",
        "fetch('/api/paper-analysis',": "fetch('./api/paper-analysis.json',",
        "fetch('/api/analysis-cases',": "fetch('./api/analysis-cases.json',",
        "fetch('/api/research-questions',": "fetch('./api/research-questions.json',",
        "fetch('/api/benchmark-tasks',": "fetch('./api/benchmark-tasks.json',",
        "fetch(`/api/experiments/${task}/${expId}`)": (
            "fetch(`./api/experiments/${encodeURIComponent(task)}/"
            "${encodeURIComponent(expId)}/detail.json`)"
        ),
        "fetch(`/api/experiments/${task}/${expId}/iterations/${n}`)": (
            "fetch(`./api/experiments/${encodeURIComponent(task)}/"
            "${encodeURIComponent(expId)}/iterations/${n}.json`)"
        ),
        "fetch(`/api/experiments/${task}/${expId}/diff/${n}`)": (
            "fetch(`./api/experiments/${encodeURIComponent(task)}/"
            "${encodeURIComponent(expId)}/diff/${n}.json`)"
        ),
        "fetch(`/api/experiments/${task}/${expId}/curves/${this.selectedIter}`)": (
            "fetch(`./api/experiments/${encodeURIComponent(task)}/"
            "${encodeURIComponent(expId)}/curves/${this.selectedIter}.json`)"
        ),
        "fetch(`/api/experiments/${task}/${expId}/code/${srcIter}/${encodeURIComponent(path)}`)": (
            "fetch(`./api/experiments/${encodeURIComponent(task)}/"
            "${encodeURIComponent(expId)}/code/${srcIter}/${encodeURIComponent(path)}.json`)"
        ),
        "new EventSource('/api/live')": "new EventSource('./api/live')",
    }
    for old, new in replacements.items():
        html = html.replace(old, new)

    html = html.replace(
        "    connectSSE() {\n"
        "      try {\n",
        "    connectSSE() {\n"
        "      if (window.FARBENCH_STATIC_DASHBOARD) {\n"
        "        this.liveStatus = 'snapshot';\n"
        "        return;\n"
        "      }\n"
        "      try {\n",
        1,
    )
    return html


def _load_dashboard_endpoints(experiments_dir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT))
    from gui.dashboard_api import create_dashboard_router  # noqa: PLC0415

    router = create_dashboard_router(experiments_dir=str(experiments_dir))
    return {route.path: route.endpoint for route in router.routes}


def _clean_output_dir(out_dir: Path) -> None:
    resolved = out_dir.resolve()
    unsafe = {Path("/").resolve(), REPO_ROOT.resolve(), REPO_ROOT.parent.resolve()}
    if resolved in unsafe:
        raise SystemExit(f"Refusing to clean unsafe output directory: {resolved}")
    if out_dir.exists():
        shutil.rmtree(out_dir)


def export_static_dashboard(
    *,
    experiments_dir: Path,
    out_dir: Path,
    clean: bool,
    include_experiment_details: bool,
    include_unscored: bool,
) -> None:
    if not experiments_dir.is_dir():
        raise SystemExit(f"Experiments directory not found: {experiments_dir}")

    if clean:
        _clean_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    api_dir = out_dir / "api"
    api_dir.mkdir(parents=True, exist_ok=True)

    endpoints = _load_dashboard_endpoints(experiments_dir)

    exported: dict[str, str] = {}
    experiments: list[dict[str, Any]] | None = None
    excluded_unscored = 0
    excluded_unscored_tasks = 0
    for route_path, filename in TOP_LEVEL_ENDPOINTS.items():
        endpoint = endpoints.get(route_path)
        if endpoint is None:
            raise SystemExit(f"Dashboard endpoint missing: {route_path}")
        data = endpoint(task=None) if route_path == "/api/experiments" else endpoint()
        if route_path == "/api/experiments" and isinstance(data, list):
            if include_unscored:
                experiments = data
            else:
                experiments = [row for row in data if row.get("scored")]
                excluded_unscored = len(data) - len(experiments)
                data = experiments
        elif route_path == "/api/benchmark-tasks" and isinstance(data, list) and not include_unscored:
            filtered_tasks = [row for row in data if row.get("scored")]
            excluded_unscored_tasks = len(data) - len(filtered_tasks)
            data = filtered_tasks
        _write_json(api_dir / filename, data)
        exported[route_path] = f"api/{filename}"
        if route_path == "/api/experiments":
            experiments = data if isinstance(data, list) else []

    detail_count = 0
    iteration_detail_count = 0
    diff_count = 0
    curve_count = 0
    code_file_count = 0
    omitted_code_file_count = 0
    if include_experiment_details and experiments:
        detail_endpoint = endpoints.get("/api/experiments/{task}/{exp_id}")
        iteration_endpoint = endpoints.get("/api/experiments/{task}/{exp_id}/iterations/{n}")
        diff_endpoint = endpoints.get("/api/experiments/{task}/{exp_id}/diff/{n}")
        code_endpoint = endpoints.get("/api/experiments/{task}/{exp_id}/code/{n}/{path:path}")
        curves_endpoint = endpoints.get("/api/experiments/{task}/{exp_id}/curves/{n}")
        if not all((detail_endpoint, iteration_endpoint, diff_endpoint, code_endpoint, curves_endpoint)):
            raise SystemExit("Dashboard detail endpoints missing")
        for exp in experiments:
            task = str(exp.get("task") or "")
            exp_id = str(exp.get("id") or "")
            if not task or not exp_id:
                continue
            try:
                detail = detail_endpoint(task=task, exp_id=exp_id)
            except Exception as exc:  # noqa: BLE001
                detail = {"error": f"{type(exc).__name__}: {exc}", "task": task, "id": exp_id}
            detail_path = (
                api_dir
                / "experiments"
                / quote(task, safe="")
                / quote(exp_id, safe="")
                / "detail.json"
            )
            _write_json(detail_path, detail)
            detail_count += 1

            trajectory = detail.get("trajectory") if isinstance(detail, dict) else None
            if not isinstance(trajectory, list):
                continue
            exp_api_dir = detail_path.parent
            # Carry-forward state: path -> iteration whose code/ snapshot holds
            # its latest content. Must be processed in iteration order.
            latest_src: dict[str, int] = {}
            ordered_rows = sorted(
                (r for r in trajectory if _iteration_number(r) is not None),
                key=_iteration_number,
            )
            for row in ordered_rows:
                n = _iteration_number(row)
                if n is None:
                    continue

                iter_detail = _safe_call(iteration_endpoint, task=task, exp_id=exp_id, n=n)
                diff_data = _safe_call(diff_endpoint, task=task, exp_id=exp_id, n=n)
                curves = _safe_call(curves_endpoint, task=task, exp_id=exp_id, n=n)

                candidates = _candidate_files(iter_detail, row, diff_data)
                exported_files: list[str] = []
                for path in candidates[:MAX_FILES_PER_ITERATION]:
                    if _is_binary_artifact(path):
                        continue
                    code_data = _safe_call(code_endpoint, task=task, exp_id=exp_id, n=n, path=path)
                    if not isinstance(code_data, dict) or code_data.get("error"):
                        continue
                    _write_json(
                        exp_api_dir / "code" / str(n) / f"{quote(path, safe='')}.json",
                        _slim_code_file(code_data),
                    )
                    exported_files.append(path)
                    latest_src[path] = n  # this iteration now holds path's content
                    code_file_count += 1

                omitted_files = max(len(candidates) - len(exported_files), 0)
                omitted_code_file_count += omitted_files

                _write_json(
                    exp_api_dir / "iterations" / f"{n}.json",
                    _slim_iteration_detail(iter_detail, latest_src, omitted_files),
                )
                _write_json(exp_api_dir / "diff" / f"{n}.json", _slim_diff(diff_data))
                _write_json(exp_api_dir / "curves" / f"{n}.json", _slim_curves(curves))
                iteration_detail_count += 1
                diff_count += 1
                curve_count += 1

    dashboard_html = (REPO_ROOT / "gui" / "dashboard.html").read_text(encoding="utf-8")
    (out_dir / "index.html").write_text(_patch_dashboard_html(dashboard_html), encoding="utf-8")
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")

    # Copy static figures (served at /figs by the live API; relative ./figs here).
    figs_src = REPO_ROOT / "gui" / "figs"
    if figs_src.is_dir():
        shutil.copytree(figs_src, out_dir / "figs", dirs_exist_ok=True)

    _write_json(
        api_dir / "export-meta.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiments_dir": str(experiments_dir),
            "experiment_count": len(experiments or []),
            "excluded_unscored_experiments": excluded_unscored,
            "excluded_unscored_tasks": excluded_unscored_tasks,
            "detail_count": detail_count,
            "iteration_detail_count": iteration_detail_count,
            "diff_count": diff_count,
            "curve_count": curve_count,
            "code_file_count": code_file_count,
            "omitted_code_file_count": omitted_code_file_count,
            "include_experiment_details": include_experiment_details,
            "include_unscored": include_unscored,
            "endpoints": exported,
            "notes": [
                "Static snapshot for GitHub Pages.",
                "Live SSE is disabled in static snapshots.",
                "Per-iteration details, diffs, training curves, and agent-touched files are exported with size caps.",
                "Full workspace snapshots are not exported.",
            ],
        },
    )

    print(f"Exported static dashboard to: {out_dir}")
    print(f"Top-level experiments: {len(experiments or [])}")
    if excluded_unscored:
        print(f"Excluded unscored experiments: {excluded_unscored}")
    print(f"Experiment detail JSON files: {detail_count}")
    print(f"Iteration detail JSON files: {iteration_detail_count}")
    print(f"Diff JSON files: {diff_count}")
    print(f"Curve JSON files: {curve_count}")
    print(f"Touched code files: {code_file_count}")
    if omitted_code_file_count:
        print(f"Omitted touched code files: {omitted_code_file_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path(os.environ.get("FARBENCH_STATIC_EXPERIMENTS_DIR", DEFAULT_EXPERIMENTS_DIR)),
        help=f"Experiment records directory (default: {DEFAULT_EXPERIMENTS_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(os.environ.get("FARBENCH_STATIC_OUT", DEFAULT_OUTPUT_DIR)),
        help=f"Output directory for GitHub Pages files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory before exporting.",
    )
    parser.add_argument(
        "--no-experiment-details",
        action="store_true",
        help="Export only top-level dashboard JSON, not per-run detail JSON.",
    )
    parser.add_argument(
        "--include-unscored",
        action="store_true",
        help="Include helper/unscored experiment directories in experiments.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_static_dashboard(
        experiments_dir=args.experiments_dir.resolve(),
        out_dir=args.out.resolve(),
        clean=args.clean,
        include_experiment_details=not args.no_experiment_details,
        include_unscored=args.include_unscored,
    )


if __name__ == "__main__":
    main()
