"""Runtime asset and output path resolution for the standalone RGB-D app.

Large models, captures, and reconstruction outputs live outside the source
repository. The runtime root can be overridden with ``--runtime`` or the
``HEADRGBD_RUNTIME_DIR`` environment variable.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_DIR = PROJECT_ROOT.parent / "withB-runtime"
ASSET_MANIFEST = PROJECT_ROOT / "config" / "assets.json"
ASSET_MANIFEST_SCHEMA = "head_rgbd.assets.v1"


@dataclass(frozen=True)
class AssetSpec:
    name: str
    relative_path: str
    sha256: str
    purpose: str

    def path(self, paths: "AssetPaths") -> Path:
        return paths.assets / self.relative_path


@dataclass(frozen=True)
class AssetPaths:
    runtime: Path
    assets: Path
    flame_dir: Path
    uniface_dir: Path
    tufa_dir: Path
    yunet_model: Path

    @property
    def flame_model(self) -> Path:
        return self.flame_dir / "flame2023.pkl"

    @property
    def flame_static(self) -> Path:
        return self.flame_dir / "flame_static_embedding.pkl"

    @property
    def flame_dynamic(self) -> Path:
        return self.flame_dir / "flame_dynamic_embedding.npy"

    @property
    def tufa_weight(self) -> Path:
        return self.tufa_dir / "Face_prompt_base.pth"

    @property
    def tufa_prompt(self) -> Path:
        return self.tufa_dir / "Prompt" / "shape_68.npz"

    def required(self) -> tuple[Path, ...]:
        return (
            self.flame_model,
            self.flame_static,
            self.flame_dynamic,
            self.yunet_model,
            self.uniface_dir / "parsing_resnet34.onnx",
            self.tufa_weight,
            self.tufa_prompt,
        )


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_asset_manifest(path: str | Path | None = None) -> tuple[AssetSpec, ...]:
    manifest_path = Path(path or ASSET_MANIFEST).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != ASSET_MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported asset manifest schema: {payload.get('schema')!r}"
        )
    specs: list[AssetSpec] = []
    names: set[str] = set()
    for raw in payload.get("assets", []):
        spec = AssetSpec(
            name=str(raw["name"]),
            relative_path=str(raw["path"]).replace("\\", "/"),
            sha256=str(raw["sha256"]).lower(),
            purpose=str(raw.get("purpose", "")),
        )
        relative = Path(spec.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"asset path must stay under assets/: {spec.relative_path}")
        if spec.name in names:
            raise ValueError(f"duplicate asset name: {spec.name}")
        if len(spec.sha256) != 64 or any(c not in "0123456789abcdef" for c in spec.sha256):
            raise ValueError(f"invalid SHA256 for asset {spec.name}")
        names.add(spec.name)
        specs.append(spec)
    if not specs:
        raise ValueError("asset manifest contains no assets")
    return tuple(specs)


def resolve_runtime_dir(explicit: str | Path | None = None) -> Path:
    value = explicit or os.environ.get("HEADRGBD_RUNTIME_DIR") or DEFAULT_RUNTIME_DIR
    return Path(value).expanduser().resolve()


def asset_paths(runtime: str | Path | None = None) -> AssetPaths:
    root = resolve_runtime_dir(runtime)
    assets = root / "assets"
    return AssetPaths(
        runtime=root,
        assets=assets,
        flame_dir=assets / "flame",
        uniface_dir=assets / "uniface",
        tufa_dir=assets / "tufa",
        yunet_model=assets / "models" / "yunet" / "face_detection_yunet_2023mar.onnx",
    )


def configure_environment(runtime: str | Path | None = None) -> AssetPaths:
    """Set every model lookup variable before importing heavyweight modules."""
    paths = asset_paths(runtime)
    os.environ["HEADRGBD_RUNTIME_DIR"] = str(paths.runtime)
    os.environ["HEADRGBD_ASSET_ROOT"] = str(paths.assets)
    os.environ["UNIFACE_CACHE_DIR"] = str(paths.uniface_dir)
    os.environ["HEADRGBD_YUNET_MODEL"] = str(paths.yunet_model)
    os.environ["HEADRGBD_FLAME_DIR"] = str(paths.flame_dir)
    os.environ["HEADRGBD_TUFA_WEIGHT"] = str(paths.tufa_weight)
    os.environ["HEADRGBD_TUFA_PROMPT"] = str(paths.tufa_prompt)
    return paths


def missing_assets(paths: AssetPaths | None = None) -> list[str]:
    paths = paths or asset_paths()
    return [
        str(spec.path(paths))
        for spec in load_asset_manifest()
        if not spec.path(paths).is_file()
    ]


def verify_assets(
    paths: AssetPaths | None = None,
    manifest_path: str | Path | None = None,
) -> list[dict]:
    """Verify every runtime asset against the checked-in immutable manifest."""
    paths = paths or asset_paths()
    result = []
    for spec in load_asset_manifest(manifest_path):
        path = spec.path(paths)
        row = {
            "name": spec.name,
            "path": str(path),
            "expected_sha256": spec.sha256,
            "purpose": spec.purpose,
            "exists": path.is_file(),
            "actual_sha256": None,
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "status": "missing",
        }
        if path.is_file():
            actual = sha256_file(path)
            row["actual_sha256"] = actual
            row["status"] = "ok" if actual == spec.sha256 else "sha256_mismatch"
        result.append(row)
    return result


def runtime_layout(runtime: str | Path | None = None) -> dict[str, Path]:
    root = resolve_runtime_dir(runtime)
    return {
        "runtime": root,
        "env": root / "env",
        "assets": root / "assets",
        "captures": root / "captures",
        "outputs": root / "outputs",
        "logs": root / "logs",
    }


def ensure_runtime_layout(runtime: str | Path | None = None) -> dict[str, Path]:
    layout = runtime_layout(runtime)
    for name, path in layout.items():
        if name != "runtime":
            path.mkdir(parents=True, exist_ok=True)
    return layout


def create_exclusive_dir(path: str | Path) -> Path:
    """Create a run directory without ever reusing or overwriting prior output."""
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def write_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def existing_only(paths: Iterable[str | Path]) -> list[Path]:
    return [Path(path) for path in paths if Path(path).exists()]
