import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import submission_file as sf  # noqa: E402


DIGEST = "6536d343785c1ea489af5743d9c66934401233521a4dff1afa950494f6723406"


def test_digest_passes_through():
    assert sf.normalise(f"@sha256:{DIGEST}") == f"@sha256:{DIGEST}"


def test_bare_hex_is_accepted():
    assert sf.normalise(DIGEST) == f"@sha256:{DIGEST}"
    assert sf.normalise(f"sha256:{DIGEST}") == f"@sha256:{DIGEST}"


def test_registry_path_is_stripped():
    ref = f"nemovaij2026.cr.cloud.ru/green-vla:iter1@sha256:{DIGEST}"
    assert sf.normalise(ref) == f"@sha256:{DIGEST}"


def test_garbage_is_rejected():
    with pytest.raises(SystemExit):
        sf.normalise("green-vla:latest")


def test_check_rejects_a_tag_with_a_digest(tmp_path):
    path = tmp_path / "submission.txt"
    path.write_text(f"registry.io/green-vla:v1@sha256:{DIGEST}\n")
    assert sf.check(path, allow_placeholder=False) == 1


def test_check_rejects_multiple_lines(tmp_path):
    path = tmp_path / "submission.txt"
    path.write_text(f"@sha256:{DIGEST}\n@sha256:{'a' * 64}\n")
    assert sf.check(path, allow_placeholder=False) == 1


def test_check_accepts_the_shipped_file():
    assert sf.check(ROOT / "submission" / "submission.txt", allow_placeholder=False) == 0


def test_make_writes_a_single_line(tmp_path):
    out = tmp_path / "s.txt"
    assert sf.make(f"@sha256:{DIGEST}", out, force=False) == 0
    assert out.read_text() == f"@sha256:{DIGEST}\n"


def test_make_refuses_to_overwrite_a_submitted_digest(tmp_path):
    out = tmp_path / "s.txt"
    sf.make(f"@sha256:{DIGEST}", out, force=False)
    other = "@sha256:" + "b" * 64
    with pytest.raises(SystemExit):
        sf.make(other, out, force=False)
    assert sf.make(other, out, force=True) == 0


def test_cli_check_on_the_shipped_file():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "submission_file.py"), "check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "@sha256:" in result.stdout
