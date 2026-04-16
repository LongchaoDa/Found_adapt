import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import toml


@dataclass(frozen=True)
class RepoContext:
    script_dir: Path
    project_dir: Path
    repo_root: Path
    data_root: Path
    param_path: Path


def resolve_repo_context(anchor_file: str) -> RepoContext:
    script_dir = Path(anchor_file).resolve().parent
    project_dir = script_dir.parent
    repo_root = project_dir.parent
    data_root = resolve_data_root(repo_root)
    return RepoContext(
        script_dir=script_dir,
        project_dir=project_dir,
        repo_root=repo_root,
        data_root=data_root,
        param_path=project_dir / "parameters.toml",
    )


def resolve_data_root(repo_root: Path) -> Path:
    candidates = []
    env_root = os.environ.get("SIM2REAL_DATA_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            repo_root / "data" / "datacollection_mini",
            repo_root / "data" / "datacollection",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    expected = ", ".join(str(path) for path in candidates)
    raise RuntimeError(
        f"Unable to locate dataset root. Checked: {expected}. "
        "Set SIM2REAL_DATA_ROOT to override."
    )


def ensure_project_imports(anchor_file: str) -> RepoContext:
    context = resolve_repo_context(anchor_file)
    for path in (context.project_dir, context.script_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.append(path_str)
    return context


def load_repo_parameters(anchor_file: str) -> Tuple[RepoContext, Dict[str, Any]]:
    context = ensure_project_imports(anchor_file)
    parameters = toml.load(context.param_path)
    _normalize_config_paths(parameters.setdefault("config", {}), context.data_root)
    return context, parameters


def _normalize_config_paths(config: Dict[str, Any], data_root: Path) -> None:
    for key, value in list(config.items()):
        if not isinstance(value, str):
            continue
        rewritten = _rewrite_data_path(value, data_root)
        if rewritten is not None:
            config[key] = rewritten


def _rewrite_data_path(raw_value: str, data_root: Path) -> Optional[str]:
    normalized = raw_value.replace("\\", "/")
    markers = (
        "/data/datacollection_mini/",
        "/data/datacollection/",
        "data/datacollection_mini/",
        "data/datacollection/",
        "../data/datacollection_mini/",
        "../data/datacollection/",
        "datacollection_mini/",
        "datacollection/",
    )
    for marker in markers:
        if marker in normalized:
            suffix = normalized.split(marker, 1)[1].lstrip("/")
            return str((data_root / suffix).resolve())
    return None
