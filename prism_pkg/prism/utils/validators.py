"""
prism/utils/validators.py
--------------------------
Silicon-Safe parameter validation for ν-Flow 1.5.

These checks enforce the three v-Flow design rules before any
hardware backend is invoked, giving clear error messages instead
of cryptic Lava / Akida runtime failures.
"""

from __future__ import annotations
import math
from typing import Any


class SiliconSafeError(ValueError):
    """
    Raised when a ν-Flow parameter set violates a Silicon-Safe rule.

    The message explains which rule was broken and how to fix it.
    """


def validate_params(
    vth:   int,
    dv:    int,
    w_exc: int,
    w_inh: int,
    floor: int,
    hw:    dict[str, Any],
) -> None:
    """
    Run all Silicon-Safe checks.  Raises SiliconSafeError on the first
    violation found.

    Parameters
    ----------
    vth   : mapped threshold integer
    dv    : mapped decay integer (0..4095 Loihi scale)
    w_exc : mapped excitatory weight integer
    w_inh : mapped inhibitory weight integer (negative)
    floor : saturation floor value (negative int, e.g. -512)
    hw    : hardware capability dict from _HW_REGISTRY
    """
    _check_bit_range(vth, w_exc, w_inh, floor, hw)
    _check_weight_ratio(w_exc, w_inh)
    _check_recovery(dv, w_inh, w_exc)
    _check_floor_depth(w_inh, floor)


# ── Individual rule checkers ─────────────────────────────────────────

def _check_bit_range(
    vth: int, w_exc: int, w_inh: int, floor: int, hw: dict
) -> None:
    """
    Rule II guard: all values must fit within the hardware integer range.
    On Loihi 2 (v_bits=32) the safe working range is ±10,000 to
    keep the fixed_pt arithmetic well clear of overflow.
    """
    v_bits = hw.get("v_bits")
    if v_bits is None:
        return   # analog target — no bit constraint

    # vth: unsigned threshold (0..2^v_bits-1, capped 10_000)
    VTH_MAX  = min(2 ** v_bits - 1, 10_000)
    # weights/floor: signed (-(2^(v_bits-1))..2^(v_bits-1)-1, capped 10_000)
    SIGN_MAX = min(2 ** (v_bits - 1) - 1, 10_000)
    SIGN_MIN = -(SIGN_MAX + 1)

    # floor is a hardware saturation register (v_min_exp exponent), not a
    # voltage accumulator value — it intentionally lives outside the working
    # range and is validated separately in _check_floor_depth.
    for name, val, lo, hi in [
        ("vth",   vth,   0,        VTH_MAX),
        ("w_exc", w_exc, SIGN_MIN, SIGN_MAX),
        ("w_inh", w_inh, SIGN_MIN, SIGN_MAX),
    ]:
        if not (lo <= val <= hi):
            raise SiliconSafeError(
                f"[Rule II — Multi-Scale Weight Map] "
                f"'{name}' = {val} is outside the safe range "
                f"[{lo}, {hi}] for {v_bits}-bit hardware.\n"
                f"Reduce the value or increase v_bits."
            )


def _check_weight_ratio(w_exc: int, w_inh: int) -> None:
    """
    Rule II guard: inhibitory weight should be at most 20× the excitatory
    weight.  Above this ratio, even a single B-spike will drive A so far
    negative that recovery becomes impractical within a 25-step window.

    The v-Flow 1.5 spec uses a 10:1 ratio (w_exc=10, w_inh=−100).
    We allow up to 20:1 before warning.
    """
    if w_exc == 0:
        return
    ratio = abs(w_inh) / abs(w_exc)
    if ratio > 20:
        raise SiliconSafeError(
            f"[Rule II — Multi-Scale Weight Map] "
            f"Inhibitory ratio |w_inh| / w_exc = {ratio:.1f}:1 exceeds "
            f"the safe 20:1 limit.  "
            f"Current: w_exc={w_exc}, w_inh={w_inh}.\n"
            f"Reduce |w_inh| or increase w_exc."
        )


def _check_recovery(dv: int, w_inh: int, w_exc: int) -> None:
    """
    Rule III guard (Leak-Matching): the decay rate must be fast enough
    that Neuron A can recover from the inhibitory well within a
    reasonable window (≤ 25 steps).

    dv is accepted in either Loihi 12-bit scale (0..4095) or
    Akida 4-bit scale (0..15) — detected automatically by range.

    The condition derived in the v-Flow 1.5 spec:
        |w_inh| × k^recovery_steps < w_exc
    Solving for recovery_steps (ceiling):
        recovery_steps = ceil(log(w_exc / |w_inh|) / log(k))
    """
    MAX_RECOVERY_STEPS = 25

    # Detect scale: Akida uses 0..15, Loihi uses 0..4095
    dv_max = 15 if dv <= 15 else 4095
    k = (dv_max - dv) / dv_max   # fraction of voltage RETAINED per step

    # k=1 → zero decay: if crush > exc, never recovers
    crush = abs(w_inh)
    exc   = abs(w_exc) if w_exc != 0 else 1

    if k >= 1.0 and crush > exc:
        raise SiliconSafeError(
            f"[Rule III — Asymmetric Decay] "
            f"dv={dv} means zero decay (k={k:.6f}), but |w_inh|={crush} > "
            f"w_exc={exc}.  Neuron A will never recover from inhibition.\n"
            f"Increase dv (e.g. dv=500 for Loihi, dv=8 for Akida) or "
            f"reduce |w_inh|."
        )

    if k <= 0 or crush <= exc:
        return   # instant decay or shallow inhibition — always recovers fast

    try:
        steps = math.ceil(math.log(exc / crush) / math.log(k))
    except (ValueError, ZeroDivisionError):
        steps = MAX_RECOVERY_STEPS + 1

    if steps > MAX_RECOVERY_STEPS:
        raise SiliconSafeError(
            f"[Rule III — Asymmetric Decay] "
            f"With dv={dv} (k={k:.4f}) and w_inh={w_inh}, "
            f"Neuron A needs ~{steps} steps to recover — "
            f"exceeds the {MAX_RECOVERY_STEPS}-step window.\n"
            f"Increase dv (faster leak) or reduce |w_inh|.\n"
            f"Tip: target dv ≈ {_suggest_dv(crush, exc, MAX_RECOVERY_STEPS, dv_max)}."
        )


def _check_floor_depth(w_inh: int, floor: int) -> None:
    """
    Rule I guard (Saturation Constraint): the floor must be at least as
    negative as the inhibitory weight so it can actually clamp the voltage.
    If the floor is shallower than w_inh, inhibition will always exceed it
    and the floor provides no protection.
    If the floor is more than 20× deeper than w_inh, it's so deep that
    recovery becomes impractical (equivalent to no floor at all).
    """
    if floor > w_inh:
        raise SiliconSafeError(
            f"[Rule I — Saturation Constraint] "
            f"Floor ({floor}) is shallower than w_inh ({w_inh}).  "
            f"Inhibitory spikes will always push below the floor.\n"
            f"Set v_min_exp so that −2^v_min_exp ≤ {w_inh}."
        )
    if abs(floor) > 20 * abs(w_inh):
        raise SiliconSafeError(
            f"[Rule I — Saturation Constraint] "
            f"Floor ({floor}) is more than 20× deeper than w_inh ({w_inh}).  "
            f"Recovery from this floor is impractical (equivalent to no floor).\n"
            f"Set v_min_exp so that −2^v_min_exp is within 10–20× of w_inh."
        )


# ── Suggestion helper ────────────────────────────────────────────────

def _suggest_dv(crush: int, exc: int, max_steps: int) -> int:
    """
    Compute the minimum dv value (Loihi 12-bit scale) needed so that
    recovery from `crush` units of inhibition takes at most `max_steps`.
    """
    # k^max_steps = exc / crush  →  k = (exc/crush)^(1/max_steps)
    import math
    k_needed = (exc / crush) ** (1.0 / max_steps)
    dv_needed = math.ceil(4096 * (1 - k_needed))
    return max(0, min(4095, dv_needed))
