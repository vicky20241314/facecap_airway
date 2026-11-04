"""Verify completed turn-experiment artifacts without running reconstruction.

Usage: runtime/env/python.exe tools/verify_turn_run.py --run <run directory>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


STAGES = ["tracking", "keyframe_selection", "prepared_keyframes", "front_anchor",
          "joint_shared_identity", "pose_correction", "control_field", "delivery"]
DELIVERY = {"init_mesh.obj", "joint_mesh.obj", "final_mesh.obj",
            "keyframe_selection.json", "experiment_config.json", "stages.json"}


def verify(run):
    run = Path(run).resolve()
    errors = []
    result = {"run": str(run), "errors": errors, "meshes": {}}

    def check(condition, message):
        if not condition:
            errors.append(message)

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    try:
        audit = read(run / "audit" / "run_report.json")
        check(audit["status"] == "passed", "Run audit did not pass")
        check([row["name"] for row in audit["stages"]] == STAGES, "Not all eight stages ran in order")
        for row in audit["stages"]:
            check(row["status"] == "passed" and row["exit_code"] == 0, f"Failed stage: {row['name']}")
            check(Path(row["log"]).is_file(), f"Missing stage log: {row['name']}")
        result["stages"] = [{key: row.get(key) for key in ("name", "status", "exit_code", "elapsed_s")}
                            for row in audit["stages"]]
        result["elapsed_s"] = audit["elapsed_s"]
        project = Path(__file__).resolve().parents[1]
        for relative, expected in audit.get("source_hashes", {}).items():
            actual = hashlib.sha256((project / relative).read_bytes()).hexdigest()
            check(actual == expected, f"Source changed since execution: {relative}")
        delivery = run / "delivery"
        check({path.name for path in delivery.iterdir()} == DELIVERY, "Delivery must contain exactly six files")
        snapshot = read(delivery / "stages.json")
        check([row["name"] for row in snapshot["stages"]] == STAGES, "Delivery stage list is incomplete")
        for row in snapshot["stages"]:
            check(row["status"] == "passed" and row["exit_code"] == 0,
                  f"Delivery stage snapshot is stale: {row['name']}")
        for stage, sequence in (("front_anchor", "anchor_sequence.json"),
                                ("joint_shape", "sequence.json"),
                                ("pose_correction", "sequence.json"),
                                ("field", "sequence.json")):
            report = read(run / stage / "report.json")
            digest = hashlib.sha256((run / "prepared" / sequence).read_bytes()).hexdigest()
            check(report["sequence_sha256"] == digest, f"Sequence hash mismatch: {stage}")
        for filename in sorted(DELIVERY):
            if not filename.endswith(".obj"):
                continue
            mesh = trimesh.load(delivery / filename, process=False, maintain_order=True)
            check(isinstance(mesh, trimesh.Trimesh), f"Not a mesh: {filename}")
            check(len(mesh.vertices) > 0 and len(mesh.faces) > 0, f"Empty mesh: {filename}")
            check(bool(np.isfinite(mesh.vertices).all()), f"Nonfinite vertices: {filename}")
            check(bool(np.all(mesh.faces >= 0) and np.all(mesh.faces < len(mesh.vertices))),
                  f"Invalid face indices: {filename}")
            result["meshes"][filename] = {"vertices": len(mesh.vertices), "faces": len(mesh.faces),
                                           "extents_m": mesh.extents.tolist()}
        selected = read(run / "selection" / "keyframes.json")
        result["keyframes"] = len(selected["keyframes"])
        result["coverage"] = selected["coverage"]
        check((delivery / "keyframe_selection.json").read_bytes() ==
              (run / "selection" / "keyframes.json").read_bytes(), "Delivered selection differs")
        field = read(run / "field" / "report.json")
        result["loop_blend"] = field["subdivision"]["loop_blend"]
        result["maximum_absolute_displacement_mm"] = field["field"]["maximum_absolute_displacement_mm"]
        result["final_safety"] = field["field"]["final_safety"]
        result["metrics"] = field["metrics"]
        experiment = read(run / "experiment.json")
        config = read(delivery / "experiment_config.json")
        check(config["config"] == experiment["config"], "Delivered config differs from executed config")
        for role, source in experiment["inputs"].items():
            for recorded in source["files"]:
                actual = hashlib.sha256(Path(recorded["path"]).read_bytes()).hexdigest()
                check(actual == recorded["sha256"], f"Input changed since execution: {role}")
        bound = experiment["config"]["field"]["displacement_bound_mm"]
        check(result["maximum_absolute_displacement_mm"] <= bound + 1e-5, "Field exceeds displacement bound")
        reference = field["reference_frame"]
        correction = read(run / "pose_correction" / "report.json")
        index = next(i for i, row in enumerate(correction["frames"]) if row["id"] == reference)
        with np.load(run / "field" / "final_vertices.npz", allow_pickle=False) as stored:
            check(bool(np.isfinite(stored["vertices"]).all()), "Nonfinite field vertices")
            mesh = trimesh.load(delivery / "final_mesh.obj", process=False, maintain_order=True)
            check(mesh.vertices.shape == stored["vertices"][index].shape and
                  np.allclose(mesh.vertices, stored["vertices"][index], rtol=0, atol=1e-7),
                  "Delivered final mesh differs from the computed reference surface")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    result["status"] = "passed" if not errors else "failed"
    result["scope"] = "Pipeline execution and artifact integrity; independent geometric accuracy is not evaluated."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    output = args.output or args.run / "verification.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "metrics"},
                     ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
