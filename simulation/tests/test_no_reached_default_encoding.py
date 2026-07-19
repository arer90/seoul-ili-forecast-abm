"""Guard: no code path this repository executes may rely on the locale encoding.

`open()`, `read_text()`, `write_text()` and `subprocess.run(text=True)` decode and
encode with `locale.getpreferredencoding()` when no `encoding=` is given. That is
UTF-8 on the Linux and macOS runners and **cp1252 on Windows**, so the omission is
invisible on two thirds of the matrix and fatal on the third. This repository is
full of Korean, which makes every such site a live Windows failure waiting for a
file with a non-ASCII byte in it.

It has already happened twice, both found by CI on 2026-07-20:

  `test_g274_permodel_audit_fixes` read `per_model_optimize.py` (881 Korean lines)
  with a bare `read_text()` and died on Windows with UnicodeDecodeError.

  `comprehensive_eval.py` wrote its per-model reports and REPORT.md with a bare
  `write_text()`. That one is not a test problem — running R12 on Windows
  produces mis-encoded reports. `real_eval.py` had the same defect in three
  places, so P1 would have done it too.

**Why this test measures reach rather than counting source sites.** A static scan
finds 256 such calls, and bulk-editing all of them in a repository this size is
how regressions get introduced. What can actually break Windows is the subset the
code REACHES, which `PYTHONWARNDEFAULTENCODING=1` reports exactly. That subset was
21 sites; all 21 are fixed, so the bar is now zero and any new one fails here.

The files below are the ones that reached such a site. Running the whole suite
under the warning flag takes ~15 minutes, which is too slow for CI, so this
guards the paths that demonstrably exercise file and subprocess I/O.

Run standalone — macOS needs per-file pytest runs (LightGBM/OpenMP):
    .venv/bin/python -m pytest simulation/tests/test_no_reached_default_encoding.py -q
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Files measured (2026-07-20) to execute a default-encoding call site. Each was
# fixed; this list keeps the fix honest. Add a file here when it starts doing
# meaningful file or subprocess I/O.
SENTINELS = [
    "simulation/tests/test_ablation_factorial.py",
    "simulation/tests/test_abm_xprocess_determinism.py",
    "simulation/tests/test_eda_writer.py",
    "simulation/tests/test_portability_baseline.py",
    "simulation/tests/test_preproc_stable_space.py",
    "simulation/tests/test_refresh_web_data.py",
    "simulation/tests/test_g274_permodel_audit_fixes.py",
    "simulation/tests/test_phase15_16_audit_fixes.py",
]

WARNING = re.compile(r"^\s*(\S+?\.py):(\d+): EncodingWarning")


def _reached_sites(test_file: str) -> list[str]:
    """Run one test file and return OUR call sites that used the locale encoding."""
    env = {**os.environ, "PYTHONWARNDEFAULTENCODING": "1"}
    p = subprocess.run(
        [sys.executable, "-W", "always", "-m", "pytest", test_file, "-q"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env=env,
    )
    hits = []
    for line in (p.stdout or "").splitlines():
        m = WARNING.search(line)
        if not m:
            continue
        path = m.group(1)
        # Third-party packages are not ours to fix (dill, huggingface_hub,
        # matplotlib and friends all have sites of their own).
        if "site-packages" in path:
            continue
        hits.append(f"{path}:{m.group(2)}")
    return hits


@pytest.mark.parametrize("test_file", SENTINELS)
def test_no_repository_code_relies_on_the_locale_encoding(test_file):
    if not (ROOT / test_file).exists():
        pytest.skip(f"{test_file} no longer exists — update SENTINELS")
    hits = _reached_sites(test_file)
    assert not hits, (
        f"{test_file} executes {len(hits)} call site(s) with no encoding=; each is "
        f"a Windows failure waiting for a non-ASCII byte:\n  " + "\n  ".join(hits)
    )


def test_the_probe_actually_detects_a_violation():
    """Without this, a broken probe would make every assertion above vacuous."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        bait = Path(d) / "test_bait_encoding.py"
        bait.write_text(
            "from pathlib import Path\n"
            "def test_bait(tmp_path):\n"
            "    p = tmp_path / 'x.txt'\n"
            "    p.write_text('hi')          # no encoding= -> locale default\n"
            "    assert p.read_text() == 'hi'\n",
            encoding="utf-8",
        )
        env = {**os.environ, "PYTHONWARNDEFAULTENCODING": "1"}
        p = subprocess.run(
            [sys.executable, "-W", "always", "-m", "pytest", str(bait), "-q"],
            cwd=d, capture_output=True, text=True, encoding="utf-8", env=env,
        )
        assert "EncodingWarning" in (p.stdout or ""), (
            "the probe did not flag a deliberately unencoded write — "
            "PYTHONWARNDEFAULTENCODING is not taking effect and every other "
            "assertion in this file is vacuous"
        )
