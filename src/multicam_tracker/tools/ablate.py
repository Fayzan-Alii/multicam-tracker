"""Run several configs and tabulate their headline metrics as Markdown."""

from __future__ import annotations

from multicam_tracker.pipeline.runner import run

MULTIVIEW_COLUMNS = [
    ("local IDF1", ("local", "IDF1")),
    ("local IDSW", ("local", "IDSW")),
    ("ground MODA", ("ground_detection", "MODA")),
    ("ground MODP", ("ground_detection", "MODP")),
    ("ground HOTA", ("ground_tracking", "HOTA")),
    ("ground MOTA", ("ground_tracking", "MOTA")),
    ("ground IDF1", ("ground_tracking", "IDF1")),
    ("ground IDSW", ("ground_tracking", "IDSW")),
    ("cross-cam IDF1", ("cross_camera", "IDF1")),
]
MOT_COLUMNS = [(k, ("COMBINED", k)) for k in ("HOTA", "MOTA", "IDF1", "DetA", "AssA", "IDSW", "Frag")]


def _cell(metrics: dict, path) -> str:
    node = metrics
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return "-"
        node = node[key]
    if isinstance(node, int) or (isinstance(node, float) and path[-1] in ("IDSW", "Frag")):
        return str(int(node))
    return f"{100 * node:.1f}"


def ablation_table(configs, log=print) -> str:
    rows = []
    for cfg in configs:
        rows.append((cfg.name, run(cfg, log=log)["metrics"]))
    columns = MOT_COLUMNS if all("COMBINED" in m for _, m in rows) else MULTIVIEW_COLUMNS
    header = "| run | " + " | ".join(name for name, _ in columns) + " |"
    sep = "|---|" + "|".join("---:" for _ in columns) + "|"
    body = ["| " + name + " | " + " | ".join(_cell(m, path) for _, path in columns) + " |" for name, m in rows]
    return "\n".join([header, sep, *body]) + "\n"
