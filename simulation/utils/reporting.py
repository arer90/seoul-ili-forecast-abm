# -*- coding: utf-8 -*-
"""
Consolidated reporting and serialization utilities for run_*.py scripts.

This module consolidates duplicated utility functions used across:
- run_full_diagnostics.py
- run_v8_enhancements.py
- run_v9_sci_validation.py
- run_v10_model_improvements.py

Functions include:
- fmt_time(): Time formatting (seconds → h:m:s)
- phase_banner(): Progress phase display
- make_serializable(): JSON serialization helper for numpy/pandas types
- model_progress(): Progress bar display (if used)

Usage:
    from simulation.utils.reporting import fmt_time, phase_banner, make_serializable
    
    elapsed = time.time() - start_time
    print(fmt_time(elapsed))
    phase_banner("R1", "Loading Data")
    serialized = make_serializable(complex_object)
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path


def fmt_time(seconds: float) -> str:
    """
    Convert seconds to human-readable time format.
    
    Args:
        seconds: Elapsed time in seconds (float)
        
    Returns:
        Formatted string like "1h23m45s" or "5m30s" or "42s"
        
    Examples:
        >>> fmt_time(0)
        '0s'
        >>> fmt_time(65)
        '1m5s'
        >>> fmt_time(3665)
        '1h1m5s'
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    
    if h > 0:
        return f"{h}h{m}m{s}s"
    elif m > 0:
        return f"{m}m{s}s"
    else:
        return f"{s}s"


def phase_banner(label: str, title: str, elapsed_seconds: float = None, 
                 use_korean: bool = False) -> None:
    """
    Print a progress phase banner to stdout/log.
    
    Args:
        label: Pipeline phase label (e.g., "R1", "v8")
        title: Descriptive title (e.g., "Loading Data")
        elapsed_seconds: Optional elapsed time in seconds (auto-calculated if None)
        use_korean: If True, use Korean formatting (한글 "경과"); else English

    Examples:
        >>> phase_banner("R1", "Data Loading")

          ==================================================
          R1: Data Loading
          경과: 5m30s
          ==================================================
    """
    try:
        import logging
        log = logging.getLogger(__name__)
    except Exception:
        log = None  # logging unavailable — fall back to None
    
    elapsed_str = ""
    if elapsed_seconds is not None:
        elapsed_str = fmt_time(elapsed_seconds)
        elapsed_label = "경과" if use_korean else "elapsed"
        elapsed_str = f"{elapsed_label}: {elapsed_str}"
    
    separator = "=" * 58
    
    output = f"\n  {separator}\n  {label}: {title}"
    if elapsed_str:
        output += f"\n  {elapsed_str}"
    output += f"\n  {separator}\n"
    
    if log:
        log.info("")
        for line in output.strip().split("\n"):
            log.info(f"  {line}")
        log.info("")
    else:
        print(output)


def make_serializable(obj, handle_dataframes: bool = True, 
                     handle_numpy_str: bool = True) -> object:
    """
    Recursively convert numpy/pandas objects to JSON-serializable Python types.
    
    Handles conversion of:
    - numpy scalar types (int64, float32, etc.) → Python int/float
    - numpy arrays → lists
    - numpy bool_ → bool
    - pandas DataFrame → list of dicts (default) or to_dict()
    - pandas Series → dict
    - Complex nested structures (dicts, lists, tuples)
    - numpy.str_ and other scalar types with .item()
    
    Args:
        obj: Object to serialize (can be nested dict/list/numpy/pandas)
        handle_dataframes: If True, convert DataFrame to records; else skip
        handle_numpy_str: If True, convert numpy.str_ to str; else skip
        
    Returns:
        JSON-serializable version of obj
        
    Examples:
        >>> import numpy as np
        >>> d = {"x": np.int64(42), "arr": np.array([1, 2, 3])}
        >>> make_serializable(d)
        {'x': 42, 'arr': [1, 2, 3]}
        
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 2], "b": [3.5, 4.5]})
        >>> make_serializable(df)
        [{'a': 1, 'b': 3.5}, {'a': 2, 'b': 4.5}]
    """
    # DataFrame → list of dicts (records)
    if handle_dataframes and isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    
    # Series → dict
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    
    # Recursive dict
    if isinstance(obj, dict):
        return {str(k): make_serializable(v, handle_dataframes, handle_numpy_str) 
                for k, v in obj.items()}
    
    # Recursive list/tuple
    if isinstance(obj, (list, tuple)):
        return [make_serializable(v, handle_dataframes, handle_numpy_str) for v in obj]
    
    # numpy integer types
    if isinstance(obj, (np.integer,)):
        return int(obj)
    
    # numpy float types
    if isinstance(obj, (np.floating,)):
        return float(obj)
    
    # numpy bool
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    
    # numpy ndarray
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # numpy string
    if handle_numpy_str and isinstance(obj, (np.str_,)):
        return str(obj)
    
    # Generic numpy scalar with .item() method
    if hasattr(obj, "item") and hasattr(obj, "dtype"):
        try:
            return obj.item()
        except (ValueError, TypeError):
            pass
    
    # Return as-is
    return obj


def save_results_json(data: dict, filepath: str, encoding: str = "utf-8", 
                     ensure_ascii: bool = False, indent: int = 2) -> Path:
    """
    Save a dictionary to JSON file with automatic serialization.
    
    Automatically calls make_serializable() before writing to handle
    numpy/pandas types. Creates parent directories if needed.
    
    Args:
        data: Dictionary to save
        filepath: Target file path (str or Path)
        encoding: File encoding (default: utf-8)
        ensure_ascii: JSON ensure_ascii parameter (default: False for Korean)
        indent: JSON indentation spaces (default: 2)
        
    Returns:
        Path object of saved file
        
    Examples:
        >>> import numpy as np
        >>> results = {"best_score": np.float32(0.95), "models": [1, 2]}
        >>> save_results_json(results, "output.json")
        PosixPath('output.json')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    serialized = make_serializable(data)
    
    with open(filepath, "w", encoding=encoding) as f:
        json.dump(serialized, f, ensure_ascii=ensure_ascii, indent=indent, default=str)
    
    return filepath


# Legacy alias for backward compatibility with run_v8//v10
def _ser(obj):
    """
    Backward-compatible alias for make_serializable().
    
    Used in run_v8_enhancements.py, run_v9_sci_validation.py, run_v10_model_improvements.py
    """
    return make_serializable(obj, handle_dataframes=True, handle_numpy_str=True)


# Legacy alias for backward compatibility with run_full_diagnostics.py
def _make_serializable(obj):
    """
    Backward-compatible alias for make_serializable().
    
    Used in run_full_diagnostics.py
    """
    return make_serializable(obj, handle_dataframes=False, handle_numpy_str=False)
