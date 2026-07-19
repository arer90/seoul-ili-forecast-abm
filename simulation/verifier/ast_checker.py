"""
simulation.verifier.ast_checker
================================
AST matcher for FORBIDDEN_PATTERNS (§9.5, RECOMMENDED_PIPELINE.md ).

Detects banned idioms that historically caused silent data leakage or
pipeline failures:

 1. `sqlite3.connect(` outside simulation.database → use safe_connect
 2. `np.random.seed(` in top-level code → use per-run Generator
 3. `multiprocessing.Process` → use subprocess.Popen
 4. `.fit_transform(X).` on full dataset before split → leakage
 5. `n_jobs=-1` → cap at 2
 6. Bare `except:` without context → use except Exception

Usage::

 from simulation.verifier import AstChecker
 checker = AstChecker
 report = checker.scan_path("simulation/models/runner.py")
 if report.n_violations:
 raise VerifierError(report.summary)

Wraps Python's `ast` module — no external dependencies.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Forbidden patterns (6 patterns, per §9.5)
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Pattern:
    name: str
    description: str
    severity: str  # 'fail' / 'warn'


FORBIDDEN_PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        name="sqlite3_connect_bypass",
        description="sqlite3.connect() bypasses safe_connect quick_check. "
                    "Use `from simulation.database import safe_connect`.",
        severity="fail",
    ),
    Pattern(
        name="global_np_random_seed",
        description="np.random.seed() at module import time pollutes other "
                    "processes. Use np.random.default_rng(seed) inside a function.",
        severity="warn",
    ),
    Pattern(
        name="multiprocessing_process",
        description="multiprocessing.Process fork/spawn conflicts with "
                    "SQLite WAL + torch. Use subprocess.Popen.",
        severity="fail",
    ),
    Pattern(
        name="n_jobs_minus_one",
        description="n_jobs=-1 oversubscribes on 8-core laptops. Cap at n_jobs=2.",
        severity="warn",
    ),
    Pattern(
        name="fit_transform_before_split",
        description="Calling fit_transform on the full dataset leaks test "
                    "statistics into training. Fit on train only.",
        severity="fail",
    ),
    Pattern(
        name="bare_except",
        description="Bare `except:` swallows KeyboardInterrupt and masks bugs. "
                    "Use `except Exception as e:`.",
        severity="warn",
    ),
)


# ══════════════════════════════════════════════════════════════════════════
# Violation / Report
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class Violation:
    pattern: str
    severity: str
    file: str
    line: int
    snippet: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.pattern} @ {self.file}:{self.line} — {self.snippet}"


@dataclass
class ScanReport:
    violations: list[Violation] = field(default_factory=list)
    files_scanned: int = 0

    @property
    def n_violations(self) -> int:
        return len(self.violations)

    @property
    def n_fail(self) -> int:
        return sum(1 for v in self.violations if v.severity == "fail")

    @property
    def n_warn(self) -> int:
        return sum(1 for v in self.violations if v.severity == "warn")

    def summary(self) -> str:
        lines = [
            f"=== AST Scan Report ({self.files_scanned} files) ===",
            f"  FAIL: {self.n_fail}  WARN: {self.n_warn}",
        ]
        for v in self.violations:
            lines.append(f"  {v}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# AstChecker — visits every AST node
# ══════════════════════════════════════════════════════════════════════════
# Methods whose semantics guarantee the input is train-only. `fit_transform`
# inside these is NOT leakage — it's the definition of "fit on training data".
# Matching is by prefix, so `fit`, `_fit_one_fold`, `pretrain_encoder`, etc.
# all qualify.
_TRAIN_SCOPE_PREFIXES: tuple[str, ...] = (
    "fit",         # sklearn-style .fit() / .fit_one(...)
    "_fit",        # private fit helpers
    "train",       # .train(), train_epoch(...)
    "_train",      # private train helpers
    "pretrain",    # transfer-learning pretrain
    "_pretrain",
    "finetune",    # fine-tuning on target domain
    "_finetune",
    "partial_fit",
    "_partial_fit",
)


class _Visitor(ast.NodeVisitor):
    def __init__(self, filepath: str, source_lines: list[str]):
        self.filepath = filepath
        self.source_lines = source_lines
        self.violations: list[Violation] = []
        # Stack of enclosing function names — top = innermost. Used to exempt
        # `fit_transform` calls inside fit/train/pretrain methods.
        self._func_stack: list[str] = []

    # ── Track enclosing function scope ───────────────────────────────
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._func_stack.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self._func_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._func_stack.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self._func_stack.pop()

    def _in_train_scope(self) -> bool:
        """True if the innermost enclosing function is a fit/train/pretrain
        method — fit_transform on its arguments is train-only by contract."""
        if not self._func_stack:
            return False
        name = self._func_stack[-1]
        return any(name.startswith(p) for p in _TRAIN_SCOPE_PREFIXES)

    def _snippet(self, node: ast.AST) -> str:
        lineno = getattr(node, "lineno", 1)
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()[:120]
        return "<unavailable>"

    def _add(self, pattern_name: str, node: ast.AST) -> None:
        pat = next((p for p in FORBIDDEN_PATTERNS if p.name == pattern_name), None)
        if pat is None:
            return
        self.violations.append(Violation(
            pattern=pattern_name,
            severity=pat.severity,
            file=self.filepath,
            line=getattr(node, "lineno", 0),
            snippet=self._snippet(node),
        ))

    # ── 1. sqlite3.connect( — allowed only inside simulation.database ──
    #     and inside simulation/tests/ (test fixtures build minimal DBs).
    def visit_Call(self, node: ast.Call) -> None:
        normalized = self.filepath.replace("\\", "/")
        sqlite_exempt = (
            "simulation/database" in normalized
            or "simulation/tests" in normalized
        )
        if _is_attr_call(node, "sqlite3", "connect") and not sqlite_exempt:
            self._add("sqlite3_connect_bypass", node)

        # 2. np.random.seed(...)
        if _is_nested_attr_call(node, ("np", "random", "seed")) or \
           _is_nested_attr_call(node, ("numpy", "random", "seed")):
            self._add("global_np_random_seed", node)

        # 3. multiprocessing.Process(...)
        if _is_attr_call(node, "multiprocessing", "Process") or \
                _is_name_call(node, "Process"):
            # Only flag `Process` when imported from multiprocessing
            if _flag_is_multiprocessing_process(node):
                self._add("multiprocessing_process", node)

        # 4. n_jobs=-1
        for kw in node.keywords:
            if kw.arg == "n_jobs" and _is_neg_one_literal(kw.value):
                self._add("n_jobs_minus_one", node)

        # 5. fit_transform — heuristic with two exemptions:
        #    (a) Enclosing function is a fit/train/pretrain/finetune method
        #        (its argument is train-only by API contract).
        #    (b) Argument name contains "train" / "tr" / "fit_" / "fit[".
        if isinstance(node.func, ast.Attribute) and node.func.attr == "fit_transform":
            if not self._in_train_scope():
                arg_name = _first_arg_name(node)
                train_arg = arg_name and any(
                    s in arg_name for s in ("train", "tr", "fit_", "fit[")
                )
                if not train_arg:
                    self._add("fit_transform_before_split", node)

        self.generic_visit(node)

    # ── 6. bare except ──
    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self._add("bare_except", node)
        self.generic_visit(node)


def _is_attr_call(node: ast.Call, mod: str, attr: str) -> bool:
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr != attr:
        return False
    val = node.func.value
    return isinstance(val, ast.Name) and val.id == mod


def _is_nested_attr_call(node: ast.Call, chain: tuple[str, ...]) -> bool:
    """Check call like chain[0].chain[1].chain[2]( ... )."""
    cur: ast.AST = node.func
    for i, name in enumerate(reversed(chain)):
        if i == 0:
            if not isinstance(cur, ast.Attribute) or cur.attr != name:
                return False
            cur = cur.value
        else:
            if isinstance(cur, ast.Name):
                return cur.id == name and i == len(chain) - 1
            if not isinstance(cur, ast.Attribute) or cur.attr != name:
                return False
            cur = cur.value
    return False


def _is_name_call(node: ast.Call, name: str) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id == name


def _flag_is_multiprocessing_process(node: ast.Call) -> bool:
    """Conservative: only flag Process(...) if it's a direct module.Attribute call."""
    return _is_attr_call(node, "multiprocessing", "Process") or \
        _is_attr_call(node, "mp", "Process")


def _is_neg_one_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        op = node.operand
        if isinstance(op, ast.Constant) and op.value == 1:
            return True
    return False


def _first_arg_name(node: ast.Call) -> Optional[str]:
    """Best-effort name extraction for a call's first positional argument.

    Handles the common idioms that appear before `.fit_transform(...)`:
      * ``X``                   → `ast.Name`        → "x"
      * ``obj.X_train``         → `ast.Attribute`   → "x_train"
      * ``X_full[train_idx]``   → `ast.Subscript`   → "x_full[train_idx]"
      * ``X[:, cols]``          → `ast.Subscript`   → "x[...]"
      * ``np.asarray(X_train)`` → `ast.Call`        → "x_train" (recurse into
                                                      first positional arg)

    Returning a compound string (e.g. "x_full[train_idx]") lets the caller's
    substring heuristic recognise train-only intent that is encoded in the
    *index* rather than the array name — the 2026-04 false-positive in
    `combinator._ridge_holdout_score`.
    """
    if not node.args:
        return None
    return _expr_name(node.args[0])


def _expr_name(a: ast.AST) -> Optional[str]:
    """Lower-cased textual proxy for an expression — see `_first_arg_name`."""
    if isinstance(a, ast.Name):
        return a.id.lower()
    if isinstance(a, ast.Attribute):
        return a.attr.lower()
    if isinstance(a, ast.Subscript):
        base = _expr_name(a.value) or ""
        idx = _expr_name(a.slice) or ""
        # Keep the base AND the index so downstream "train" / "tr" substring
        # probes can hit either side — e.g. X_full[train_idx] → "x_full[train_idx]".
        return f"{base}[{idx}]" if idx else base or None
    if isinstance(a, ast.Tuple):
        # np-style fancy indexing: X[:, cols] → slice tuple
        parts = [p for p in (_expr_name(elt) for elt in a.elts) if p]
        return ",".join(parts) if parts else None
    if isinstance(a, ast.Slice):
        # X[train_start:train_end] — return "train_start:train_end" so "train"
        # substring fires.
        lo = _expr_name(a.lower) if a.lower else ""
        hi = _expr_name(a.upper) if a.upper else ""
        joined = f"{lo}:{hi}".strip(":")
        return joined or None
    if isinstance(a, ast.Call):
        # np.asarray(X_train) / torch.tensor(X_train) — recurse one level.
        if a.args:
            return _expr_name(a.args[0])
        return None
    return None


class AstChecker:
    """Batch AST scanner."""

    def __init__(self, patterns: Optional[Iterable[Pattern]] = None):
        self.patterns = list(patterns) if patterns is not None else list(FORBIDDEN_PATTERNS)

    def scan_source(self, source: str, filepath: str = "<string>") -> ScanReport:
        """Scan a string of Python source."""
        return self._scan(source, filepath)

    def scan_path(self, path: str | Path) -> ScanReport:
        """Scan a single .py file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        return self._scan(p.read_text(encoding="utf-8"), str(p))

    def scan_dir(
        self,
        root: str | Path,
        *,
        exclude: Iterable[str] = ("__pycache__", "_archive", "_legacy", "_past"),
    ) -> ScanReport:
        """Recursively scan a directory."""
        root = Path(root)
        report = ScanReport()
        excl = set(exclude)
        for py in root.rglob("*.py"):
            if any(part in excl for part in py.parts):
                continue
            sub = self._scan(py.read_text(encoding="utf-8"), str(py))
            report.violations.extend(sub.violations)
            report.files_scanned += sub.files_scanned
        return report

    def _scan(self, source: str, filepath: str) -> ScanReport:
        try:
            tree = ast.parse(source, filename=filepath)
        except SyntaxError as e:
            log.warning("AST parse failed on %s: %s", filepath, e)
            return ScanReport(files_scanned=1)
        visitor = _Visitor(filepath=filepath, source_lines=source.splitlines())
        visitor.visit(tree)
        return ScanReport(violations=visitor.violations, files_scanned=1)


# ══════════════════════════════════════════════════════════════════════════
# Convenience wrappers
# ══════════════════════════════════════════════════════════════════════════
def scan_file(path: str | Path) -> ScanReport:
    return AstChecker().scan_path(path)


def scan_source(src: str) -> ScanReport:
    return AstChecker().scan_source(src)
