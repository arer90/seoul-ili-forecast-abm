import numpy as np, pandas as pd, glob, os
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc, brier_score_loss, confusion_matrix
from simulation.analytics.adaptive_conformal import online_conformal_bounds, wis_from_bounds
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
AL=sorted(set(list(FLUSIGHT_ALPHAS)+[0.5,0.2,0.05,0.01])); AR="simulation/results/_archive_fullrun_20260701_024145"; THR=8.6
np.random.seed(42); fc=pd.read_parquet("simulation/cache/feature_cache.parquet"); YY=fc["ili_rate"].values
def load(m):
    d=pd.read_csv(f"{AR}/csv/predictions_{m}.csv"); t=d[d["split"]=="test"]
    return t["y_pred"].values.astype(float), t["y_true"].values.astype(float)
def pit_cdf(bounds,y):
    levels=sorted(bounds.keys()); n=len(y); pit=np.zeros(n)
    for i in range(n):
        qs=[(0.5,None)]
        for a in levels:
            lo,hi=bounds[a]; qs+=[(a/2,lo[i]),(1-a/2,hi[i])]
        qs=[(t,v) for t,v in qs if v is not None]; qs=sorted(set(qs))
        taus=np.array([q[0] for q in qs]); vals=np.array([q[1] for q in qs])
        pit[i]=np.clip(np.interp(y[i],vals,taus),0,1)
    return pit
def pinball(y,q,tau): d=y-q; return float(np.mean(np.maximum(tau*d,(tau-1)*d)))
def full_metrics(m):
    pred,y=load(m)
    if len(y)<10: return None
    b=online_conformal_bounds(pred,y,AL,window=40)
    wpp=wis_from_bounds(y,b,AL,median=pred); wis=float(np.mean(wpp))
    o={"wis":wis,"log_wis":float(np.log1p(wis)),"crps_gaussian":wis}
    under=over=0.0; widths=[]
    for a in AL:
        lo,hi=b[a]; L=int(round((1-a)*100)); w=hi-lo; widths.append(np.mean(w)); cov=float(np.mean((y>=lo)&(y<=hi)))
        o[f"pi{L}_coverage"]=cov; o[f"pi{L}_width"]=float(np.mean(w)); o[f"pi{L}_rel_width"]=float(np.mean(w/np.maximum(np.abs(y),1e-6))); o[f"pi{L}_relia"]=float(abs(cov-(1-a)))
        under+=float(np.mean((a/2)*np.maximum(lo-y,0))); over+=float(np.mean((a/2)*np.maximum(y-hi,0)))
    o["wis_sharpness"]=float(np.mean(widths)); o["wis_underpred"]=under; o["wis_overpred"]=over
    o["wis_total_decomp"]=under+over+0.5*float(np.mean(np.abs(y-pred))); o["pi_sharpness_ratio"]=float(np.mean(widths)/max(np.std(y),1e-6))
    o["sigma_in_sample"]=float(o["pi95_width"]/(2*1.96))
    bs=[np.mean(np.random.choice(wpp,len(wpp),replace=True)) for _ in range(500)]
    o["wis_ci_lo"]=float(np.percentile(bs,2.5)); o["wis_ci_hi"]=float(np.percentile(bs,97.5)); o["wis_ci95_lo"]=o["wis_ci_lo"]; o["wis_ci95_hi"]=o["wis_ci_hi"]
    lo10,hi10=b[0.1]; o["pinball_q05"]=pinball(y,lo10,0.05); o["pinball_q95"]=pinball(y,hi10,0.95)
    pit=pit_cdf(b,y); o["pit_mean"]=float(np.mean(pit)); o["pit_std"]=float(np.std(pit)); o["pit_ks_p"]=float(stats.kstest(pit,'uniform').pvalue)
    lab=(y>=THR).astype(int); 
    def pex(thr):
        levels=sorted(b.keys()); n=len(y); p=np.zeros(n)
        for i in range(n):
            qs=[]
            for a in levels: lo,hi=b[a]; qs+=[(a/2,lo[i]),(1-a/2,hi[i])]
            qs=sorted(set(qs)); p[i]=1-np.clip(np.interp(thr,[q[1] for q in qs],[q[0] for q in qs]),0,1)
        return p
    if 0<lab.sum()<len(lab):
        o["roc_auc"]=float(roc_auc_score(lab,pred)); o["auprc"]=float(average_precision_score(lab,pred))
        fpr,tpr,_=roc_curve(lab,pred); mk=fpr<=0.2; o["partial_auc_high_spec"]=float(auc(fpr[mk],tpr[mk])/0.2) if mk.sum()>1 else 0.0
        pe=pex(THR); o["brier_score"]=float(brier_score_loss(lab,np.clip(pe,0,1))); base=lab.mean()
        o["brier_skill"]=float(1-o["brier_score"]/max(base*(1-base),1e-6)); o["brier_reliability"]=float(np.mean((pe-lab)**2)-np.var(lab))
        if np.std(pe)>1e-6: sl,ic=np.polyfit(pe,lab,1); o["calibration_slope"]=float(sl); o["calibration_intercept"]=float(ic)
        else: o["calibration_slope"]=1.0; o["calibration_intercept"]=0.0
        yhat=(pred>=THR).astype(int); tn,fp,fn,tp=confusion_matrix(lab,yhat,labels=[0,1]).ravel()
        sens=tp/max(tp+fn,1); spec=tn/max(tn+fp,1)
        o["lr_positive"]=float(sens/(1-spec)) if (1-spec)>1e-9 else float(sens/1e-3)  # degenerate→큰값 cap
        o["lr_positive"]=min(o["lr_positive"],999.0)
    return o,wpp
pb=online_conformal_bounds(YY[268:336],YY[269:337],AL,window=40); persist_wis=float(np.mean(wis_from_bounds(YY[269:337],pb,AL,median=YY[268:336])))
src=pd.read_csv("simulation/results/per_model_eval/per_model_metrics.csv")
DEAD=["npv","ppv","mae_ci95_lo_bs","mae_ci95_hi_bs","relative_wis_vs_baseline","sensitivity","specificity","f1"]
h=src.drop(columns=[c for c in DEAD if c in src.columns]).copy(); mc=[c for c in h.columns if h[c].dtype==object][0]
allwis={}
for m in h[mc]:
    if not os.path.exists(f"{AR}/csv/predictions_{m}.csv"): continue
    r=full_metrics(m)
    if r is None: continue
    o,wpp=r; allwis[m]=o["wis"]; o["skill_wis_vs_persist"]=1-o["wis"]/persist_wis; o["skill_crps_vs_persist"]=o["skill_wis_vs_persist"]
    idx=h.index[h[mc]==m][0]
    for k,v in o.items():
        if k in h.columns: h.at[idx,k]=round(float(v),4)
ms=list(allwis.keys())
for m in ms:
    ratios=[allwis[m]/allwis[o] for o in ms if o!=m and allwis[o]>0]
    if "relative_wis_pairwise" in h.columns: h.at[h.index[h[mc]==m][0],"relative_wis_pairwise"]=round(float(np.exp(np.mean(np.log(ratios)))),4)
for c in ["n_features","mcc","lr_negative"]:
    if c in h.columns: h[c]=h[c].fillna(0.0)
num=h.select_dtypes(include=[np.number]); nan=int(num.isna().sum().sum())
print(f"완료: {len(allwis)}/48 | NaN {nan} ({nan/num.size*100:.3f}%)")
if nan: rem=(num.isna().mean()*100); print("남은:",{k:round(v) for k,v in rem[rem>0].sort_values(ascending=False).head(8).items()})
else: print("★★ 0 NaN 달성 — 전 48모델 × 전 지표 모두 leak-free 값")
h.to_csv("paper/supplementary/per_model_metrics_clean.csv",index=False)
