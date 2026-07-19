"""Age and occupation contact assumptions for the ABM agent kernel.

The age contact matrix is a POLYMOD-like symmetric default scaled to a
Korean 2023-24 ILI survey anchor of 4.81 contacts/day. It is an assumption
for synthetic Seoul ABM experiments, not a directly estimated Korean age
mixing matrix.
"""

from __future__ import annotations

import numpy as np


AGE_BAND_LABELS = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+"]

_POLYMOD_LIKE_7x7 = np.array(
    [
        [7.2, 5.9, 2.4, 2.1, 1.8, 1.2, 0.8],
        [5.9, 9.4, 4.2, 2.7, 2.0, 1.4, 0.9],
        [2.4, 4.2, 6.7, 4.7, 3.2, 1.8, 1.1],
        [2.1, 2.7, 4.7, 6.1, 4.3, 2.4, 1.5],
        [1.8, 2.0, 3.2, 4.3, 5.6, 3.2, 2.0],
        [1.2, 1.4, 1.8, 2.4, 3.2, 4.4, 2.8],
        [0.8, 0.9, 1.1, 1.5, 2.0, 2.8, 3.9],
    ],
    dtype=np.float64,
)

# Fixed POLYMOD-like assumption scaled to the Korean 2023-24 ILI survey
# contact anchor of 4.81 contacts/day.
CONTACT_MATRIX_7x7 = (
    _POLYMOD_LIKE_7x7 * (4.81 / float(_POLYMOD_LIKE_7x7.mean(axis=1).mean()))
).astype(np.float64)

# Fixed occupation exposure multipliers for scenario analysis.
OCCUPATION_EXPOSURE = {
    "service": 1.6,
    "essential": 1.5,
    "office": 1.0,
    "school": 1.3,
    "other": 0.8,
    "unemployed": 0.7,
}


def get_contact_rate(age_band_i: int, age_band_j: int) -> float:
    """Return the assumed contacts/day rate for an age-band pair.

    Args:
        age_band_i: Source age-band index in ``0..6``.
        age_band_j: Target age-band index in ``0..6``.

    Returns:
        Float contact-rate entry from ``CONTACT_MATRIX_7x7``.

    Raises:
        IndexError: If either age-band index is outside ``0..6``.

    Performance: O(1) time and memory.
    Side effects: none.
    Caller responsibility: pass integer age-band codes matching
        ``AGE_BAND_LABELS``.
    """
    return float(CONTACT_MATRIX_7x7[int(age_band_i), int(age_band_j)])


def get_occupation_multiplier(occupation: str) -> float:
    """Return the fixed exposure multiplier for an occupation category.

    Args:
        occupation: Occupation category name. Known values are the keys of
            ``OCCUPATION_EXPOSURE``.

    Returns:
        Exposure multiplier, or ``1.0`` for unknown categories.

    Performance: O(1) time and memory.
    Side effects: none.
    Caller responsibility: normalize project-specific occupation labels before
        using them as policy categories when exact lookup is needed.
    """
    return float(OCCUPATION_EXPOSURE.get(str(occupation), 1.0))
