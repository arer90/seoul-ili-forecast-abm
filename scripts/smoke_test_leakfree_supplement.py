"""SMOKE TEST: uniform online-conformal per-model metrics — leak-free + complete + reproducible."""
import numpy as np, pandas as pd, glob, os, sys
from simulation.analytics.adaptive_conformal import online_conformal_bounds, wis_from_bounds
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
AL=sorted(set(list(FLUSIGHT_ALPHAS)+[0.5,0.2,0.05,0.01])); AR="simulation/results/_archive_fullrun_20260701_024145"
fails=[]
# TEST 1: leak-free (미래 관측 교란 → 과거 구간 절대 불변) — 6 모델
print("TEST 1 — leak-free (미래 y 교란 → 과거 구간 불변):")
for m in ["FusedEpi","TiRex","ARIMA","N-HiTS","Ensemble-BMA","XGBoost"]:
    d=pd.read_csv(f"{AR}/csv/predictions_{m}.csv"); t=d[d["split"]=="test"]
    pred=t["y_pred"].values.astype(float); y=t["y_true"].values.astype(float); k=40
    b1=online_conformal_bounds(pred,y,AL,window=40)
    y2=y.copy(); y2[k:]+=1e4
    b2=online_conformal_bounds(pred,y2,AL,window=40)
    a=AL[len(AL)//2]
    ok=np.allclose(b1[a][0][:k],b2[a][0][:k]) and np.allclose(b1[a][1][:k],b2[a][1][:k])
    print(f"   {m:<14} {'✅ 불변' if ok else '❌ LEAK'}"); 
    if not ok: fails.append(f"leak:{m}")
# TEST 2: 완성 CSV — NaN/inf/음수 WIS/coverage 범위
print("\nTEST 2 — 완성 매트릭스 무결성:")
h=pd.read_csv("paper/supplementary/per_model_metrics_clean.csv")
num=h.select_dtypes(include=[np.number])
nan=int(num.isna().sum().sum()); inf=int(np.isinf(num.values).sum())
print(f"   NaN={nan} {'✅' if nan==0 else '❌'} | inf={inf} {'✅' if inf==0 else '❌'} | shape={h.shape}")
if nan: fails.append("nan"); 
if inf: fails.append("inf")
if "wis" in h: 
    negw=(h["wis"]<0).sum(); print(f"   음수 WIS={negw} {'✅' if negw==0 else '❌'}")
    if negw: fails.append("negwis")
for c in ["pi95_coverage","pi50_coverage","roc_auc"]:
    if c in h: 
        bad=((h[c]<0)|(h[c]>1)).sum(); print(f"   {c} 범위[0,1] 위반={bad} {'✅' if bad==0 else '❌'}")
        if bad: fails.append(f"range:{c}")
# TEST 3: 재현성 (같은 입력 → 같은 출력)
print("\nTEST 3 — 재현성:")
d=pd.read_csv(f"{AR}/csv/predictions_FusedEpi.csv"); t=d[d["split"]=="test"]
pred=t["y_pred"].values.astype(float); y=t["y_true"].values.astype(float)
w1=np.mean(wis_from_bounds(y,online_conformal_bounds(pred,y,AL,window=40),AL,median=pred))
w2=np.mean(wis_from_bounds(y,online_conformal_bounds(pred,y,AL,window=40),AL,median=pred))
print(f"   2회 실행 동일? {w1==w2} {'✅' if w1==w2 else '❌'}")
if w1!=w2: fails.append("nondeterministic")
# TEST 4: 챔피언 sanity (WIS 상위권, coverage 명목 근처)
print("\nTEST 4 — 챔피언 sanity:")
fe=h[h.iloc[:,0]=="FusedEpi"].iloc[0]
rank=(h["wis"]<fe["wis"]).sum()+1
print(f"   FusedEpi WIS={fe['wis']:.3f} 순위={rank}/{len(h)} (상위권?{'✅' if rank<=5 else '⚠'}) | cov95={fe['pi95_coverage']:.3f} (명목0.95 근처?{'✅' if 0.8<=fe['pi95_coverage']<=1.0 else '⚠'})")
print(f"\n{'='*50}\n{'★ SMOKE TEST 전부 통과' if not fails else '❌ 실패: '+str(fails)}")
sys.exit(1 if fails else 0)
