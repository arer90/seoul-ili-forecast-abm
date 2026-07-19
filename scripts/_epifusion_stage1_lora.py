#!/usr/bin/env python
"""EpiFusion-XL Stage 1 — LoRA-adapt the TiRex (xLSTM) backbone on a leak-free multi-country
ILI corpus and test the CRITICAL GATE: does LoRA improve the 1-step POINT over zero-shot TiRex?

Assembles ONLY successful-model parts: TiRex xLSTM backbone (rolling rel-WIS #1) + LoRA adapters
(frozen base + rank-r, do-no-harm). Trains adapters on Seoul-train + clean other-country national
series (US delphi, JP jihs, + European nationals), differentiable _forecast_tensor, normalized loss.
Then rolls the LoRA-adapted TiRex 1-step over the SAME Seoul 132 test origins (weeks 205..336) and
compares point error + (Tweedie interval on top) WIS to zero-shot TiRex.

Leak-free: LoRA training uses ONLY weeks < each series' eval-test region; Seoul test weeks (269..336)
and the 132 rolling origins' own future are NEVER in training. do-no-harm gate zeros LoRA if val worse.
No live/pipeline edits (imports lora_inject + generic Tweedie funcs from _exp_crosscountry).
"""
from __future__ import annotations
import json, sqlite3, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DB = ROOT / "simulation/data/db/epi_real_seoul.db"


def load_clean(country, source):
    con = sqlite3.connect(DB)
    rows = con.execute("SELECT ili_rate FROM overseas_ili WHERE country=? AND source=? AND ili_rate IS NOT NULL "
                       "ORDER BY year,week_no", (country, source)).fetchall()
    con.close()
    return np.clip(np.array([r[0] for r in rows], float), 0.0, None)


def seoul_series():
    from scripts.nov_guard_v3 import setup
    S = setup()
    return S["yf"], S["tirex"], S["ntot"]


def train_lora(train_series, rank=4, alpha=8.0, epochs=3, lr=5e-4, stride=2, min_ctx=52, max_ctx=512):
    """LoRA-adapt TiRex on a list of (name, y, train_end) series (weeks < train_end used only)."""
    import torch
    from tirex import load_model
    from simulation.models.lora_inject import inject_lora
    model = load_model("NX-AI/TiRex", device="cpu")
    model, n_tr = inject_lora(model, rank=rank, alpha=alpha)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    trainable = [p for p in model.parameters() if p.requires_grad]
    model.train()
    n_steps = 0
    for _ep in range(epochs):
        for name, y, tend in train_series:
            gscale = max(float(np.std(y[:tend])), 1.0)
            for t in range(min_ctx, tend, stride):
                ctx_raw = y[max(0, t - max_ctx):t]
                if len(ctx_raw) < min_ctx:
                    continue
                scale = max(float(np.std(ctx_raw)), 0.5 * gscale, 1.0)
                ctx = torch.tensor(ctx_raw, dtype=torch.float32).unsqueeze(0)
                try:
                    pred = model._forecast_tensor(ctx, prediction_length=1)
                    med = pred.reshape(pred.shape[0], pred.shape[1], -1)[:, pred.shape[1] // 2, 0]
                    loss = (((med - float(y[t])) / scale) ** 2).mean()
                    if not torch.isfinite(loss):
                        continue
                    opt.zero_grad(); loss.backward()
                    # ★ TiRex input_patch_embedding produces NaN grads -> sanitize to 0 (those layers
                    #   don't update; the 60/84 stable LoRA params train). Without this, 1 step -> NaN blowup.
                    for p in trainable:
                        if p.grad is not None:
                            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); n_steps += 1
                except Exception:
                    continue
    model.eval()
    return model, n_tr, n_steps


def roll_point(model, y, idxs, max_ctx=512):
    import torch
    out = np.full(len(idxs), np.nan)
    with torch.no_grad():
        for k, t in enumerate(idxs):
            ctx = torch.tensor(y[max(0, t - max_ctx):t], dtype=torch.float32).unsqueeze(0)
            _q, mean = model.forecast(context=ctx, prediction_length=1)
            out[t if False else k] = float(np.asarray(mean).ravel()[0])
    return out


def main():
    t0 = time.time()
    import scripts._exp_crosscountry as X
    y_seoul, tirex_zs, ntot = seoul_series()
    # ---- CLEAN 3-WAY SPLIT (no overlap): TRAIN [52,165) | VAL [165,205) | TEST [205,337) ----
    T0 = 205                     # TEST start (earliest test origin)
    SEOUL_TRAIN_END = 165        # TRAIN = weeks [52,165); VAL = [165,205); TEST = [205,337)
    origins = np.arange(T0, ntot); n = len(origins); y_te = y_seoul[origins]

    # ---- training corpus: Seoul TRAIN + other-country nationals (each excl. its OWN val+test) ----
    us = load_clean("US", "delphi_national")     # 1404 wk; US val+test = last 340 -> train < 1064
    jp = load_clean("JP", "japan_jihs")          # 316 wk;  JP val+test = last 172 -> train < 144
    at = load_clean("AT", "ecdc_erviss")         # ~134 wk (European; not an eval target -> full is fine)
    train_series = [
        ("seoul_train", y_seoul, SEOUL_TRAIN_END),    # Seoul weeks < 165 (VAL+TEST fully excluded)
        ("us", us, len(us) - 340),                    # exclude US val+test tail
        ("jp", jp, len(jp) - 172),                    # exclude JP val+test tail
        ("at", at, len(at)),
    ]
    print(f"[split] Seoul  TRAIN=[52,{SEOUL_TRAIN_END})  VAL=[{SEOUL_TRAIN_END},{T0})  TEST=[{T0},{ntot})  (no overlap)")
    print(f"[corpus] series: {[(nm, tend) for nm, _, tend in train_series]}")
    # leak-free assertion: Seoul training must not reach VAL or TEST
    assert train_series[0][2] <= SEOUL_TRAIN_END <= T0 - X.K_CAL, "Seoul train overlaps VAL/TEST!"
    model, n_tr, n_steps = train_lora(train_series, rank=4, alpha=8.0, epochs=3, lr=5e-4, stride=2)
    print(f"[LoRA] {n_tr} adapter params, {n_steps} train steps, {time.time()-t0:.0f}s")

    # ---- do-no-harm on Seoul HELD-OUT val [185,205): not in LoRA training, before test -> leak-free ----
    val_idx = list(range(SEOUL_TRAIN_END, T0))
    lora_val = roll_point(model, y_seoul, val_idx)
    zs_val = tirex_zs[np.array(val_idx)]
    yv = y_seoul[np.array(val_idx)]
    mae_lora = float(np.nanmean(np.abs(lora_val - yv))); mae_zs = float(np.nanmean(np.abs(zs_val - yv)))
    use_lora = mae_lora < mae_zs
    print(f"[do-no-harm] Seoul val MAE: LoRA={mae_lora:.4f} vs zero-shot={mae_zs:.4f} -> use_lora={use_lora}")

    # ---- roll LoRA point on the 132 Seoul test origins; build TiRex_lora full array ----
    lora_test = roll_point(model, y_seoul, list(origins))
    tirex_lora = tirex_zs.copy()
    tirex_lora[origins] = lora_test if use_lora else tirex_zs[origins]

    # ---- point error on test: LoRA vs zero-shot ----
    mae_t_lora = float(np.mean(np.abs(tirex_lora[origins] - y_te)))
    mae_t_zs = float(np.mean(np.abs(tirex_zs[origins] - y_te)))

    # ---- Tweedie interval on BOTH points, DM ----
    cap = 2.0 * float(np.nanmax(y_seoul[:T0]))
    def tweedie_eval(tirex_arr):
        val = np.arange(T0 - X.K_CAL, T0); y_val = y_seoul[val]
        vw = {}
        for p in X.P_GRID:
            vqy = X.tweedie_qy(y_seoul, tirex_arr, val, p, cap); vB = X.expanding_cqr_bounds(vqy, y_val, cap)
            vw[p] = float(X.wis_of(vB, y_val, vqy[:, X.MED_COL]).mean())
        p_star = min(vw, key=vw.get)
        tqy = X.tweedie_qy(y_seoul, tirex_arr, origins, p_star, cap); tB = X.expanding_cqr_bounds(tqy, y_te, cap)
        w = X.wis_of(tB, y_te, tqy[:, X.MED_COL]); lo, hi = tB[0.05]; k = int(((y_te >= lo) & (y_te <= hi)).sum())
        return w, p_star, k
    w_zs, p_zs, k_zs = tweedie_eval(tirex_zs)
    w_lora, p_lo, k_lora = tweedie_eval(tirex_lora)
    dmp, dbar = X.dm(w_lora, w_zs)

    out = {
        "use_lora": bool(use_lora), "n_adapter_params": int(n_tr), "n_train_steps": int(n_steps),
        "seoul_val_mae_lora": round(mae_lora, 4), "seoul_val_mae_zeroshot": round(mae_zs, 4),
        "test_point_mae_lora": round(mae_t_lora, 4), "test_point_mae_zeroshot": round(mae_t_zs, 4),
        "tweedie_wis_zeroshot": round(float(w_zs.mean()), 4), "p_zeroshot": p_zs, "picp95_zs": round(k_zs/n, 3),
        "tweedie_wis_lora": round(float(w_lora.mean()), 4), "p_lora": p_lo, "picp95_lora": round(k_lora/n, 3),
        "dm_p_lora_vs_zeroshot": round(dmp, 4), "dm_meandiff": round(dbar, 4),
        "lora_beats_champion": bool(w_lora.mean() < w_zs.mean() and dmp < 0.05),
        "elapsed_s": round(time.time()-t0, 0),
    }
    (ROOT / "scripts" / "_epifusion_stage1.json").write_text(json.dumps(out, indent=2))
    print("\n=== STAGE 1 RESULT ===")
    print(json.dumps(out, indent=2))
    print("\nVERDICT:", "LoRA IMPROVES the point -> proceed to TabPFN head + full assembly"
          if out["lora_beats_champion"] else
          ("LoRA reverted (do-no-harm) = no point gain on this small corpus; EpiFusion-XL point == zero-shot TiRex"
           if not use_lora else "LoRA used but does not beat champion Tweedie (point gain didn't translate to WIS)"))


if __name__ == "__main__":
    raise SystemExit(main())
