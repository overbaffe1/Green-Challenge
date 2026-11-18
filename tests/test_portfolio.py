"""Keep the offline exhibit tied to the wire capture and numeric layout."""

import json
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from greenchallenge import layout
from tools.build_portfolio import LAB, MEDIA, ROOT
from tools.portfolio_graphics import relief_grid


def replay() -> dict:
    data = (ROOT / LAB / "replay.js").read_text(encoding="utf-8")
    return json.loads(data.split("window.GC_REPLAY = ", 1)[1].strip().removesuffix(";"))


@pytest.mark.parametrize("mode,flags", [("v18-drop", []), ("v18-keep", ["--keep-frozen"])])
def test_capture_matches_exhibited_ab_run(tmp_path: Path, mode: str, flags: list[str]):
    capture = tmp_path / "capture.json"
    subprocess.run([
        sys.executable, "tools/smoke_test.py", "--seconds", "3", "--speed", "1.8", *flags,
        "--capture", str(capture), "--telemetry", str(tmp_path / "telemetry.jsonl"),
    ], cwd=ROOT, capture_output=True, check=True, timeout=30)
    got = json.loads(capture.read_text(encoding="utf-8"))
    expected = replay()["captures"][mode]
    rows, recorded = got.pop("requests"), expected.pop("requests")
    assert got == expected
    assert got["joint_names"] == ["right_shoulder_pitch", "right_elbow_pitch"]
    assert got["model_calls"] == len(rows) == 38
    values = []
    for i, (actual, sample) in enumerate(zip(rows, recorded, strict=True)):
        assert actual["t"] == round(i * .08, 6)
        commands, original = actual.pop("commands"), sample.pop("commands")
        assert actual == sample
        assert np.asarray(commands).shape == (20, 2)
        np.testing.assert_allclose(commands, original, atol=1e-6, rtol=0)
        values.extend(commands)
    assert np.ptp(np.asarray(values)[:, 1]) > .5
    assert got["live_peak"] == (15 if flags else 12)
    assert rows[-1]["frozen_dropped"] == (0 if flags else 27)


def test_offline_zip_and_patch_panel_are_in_sync():
    with ZipFile(ROOT / MEDIA / "control-room.zip") as archive:
        assert archive.testzip() is None
        files = ("index.html", "lab.css", "lab.js", "replay.js")
        assert set(archive.namelist()) == {f"green-control-room/{name}" for name in files}
        for name in files:
            assert archive.read(f"green-control-room/{name}") == (ROOT / LAB / name).read_bytes()
    patch = replay()["patch"]
    assert len(patch["raw_names"]) == layout.RAW_STATE_DIM
    assert patch["raw_names"][28] == "right_wrist_roll"
    assert patch["raw_names"][44] == "root_yaw_global"
    assert patch["model_to_raw"] == layout.MODEL_TO_RAW_51.tolist()
    assert patch["wire_to_raw"] == layout.raw_to_wire(np.arange(layout.RAW_STATE_DIM)).astype(int).tolist()
    assert patch["delta_mask"] == layout.delta_mask().tolist()


def test_relief_is_a_closed_mesh_of_real_command_samples():
    stl = (ROOT / MEDIA / "command-relief.stl").read_bytes()
    count, = struct.unpack_from("<I", stl, 80)
    assert len(stl) == 84 + count * 50
    edges = Counter()
    vertices = set()
    orientation = Counter()
    for face in struct.iter_unpack("<12fH", stl[84:]):
        a, b, c = face[3:6], face[6:9], face[9:12]
        assert len({a, b, c}) == 3
        vertices.update((a, b, c))
        for p, q in ((a, b), (b, c), (c, a)):
            edges[tuple(sorted((p, q)))] += 1
            orientation[p, q] += 1
    assert set(edges.values()) == {2}
    assert all(orientation[q, p] == n == 1 for (p, q), n in orientation.items())
    run = replay()["captures"]["v13"]
    for i, request in enumerate(run["requests"]):
        for j, (_, elbow) in enumerate(request["commands"]):
            point = ((i + .5) * 3.5 - .9, j * 3.4, 2.2 + 22 * elbow)
            assert tuple(np.float32(point)) in vertices
    assert len(relief_grid(run)) == 38 * 3 + 1
