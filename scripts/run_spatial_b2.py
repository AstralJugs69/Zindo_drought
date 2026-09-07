"""Causal B2 temporal-factor ridge screen (Kaggle only; no Test predictions)."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.run_spatial_phase_a import _basis

ORIGINS=("2007-09","2009-01","2014-12"); RANKS=(8,16)
HYDRO=("SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t","SOIL_MOISTURE_t")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',type=Path,required=True); ap.add_argument('--run-dir',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--expected-commit',required=True); ap.add_argument('--origins',default=','.join(ORIGINS)); a=ap.parse_args()
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if head!=a.expected_commit: raise RuntimeError((head,a.expected_commit))
    a.output_dir.mkdir(parents=True,exist_ok=False); train=pd.read_csv(a.data_dir/'Train.csv'); started=time.perf_counter(); results=[]
    origins=tuple(x for x in a.origins.split(',') if x in ORIGINS)
    for origin in origins:
        op=a.run_dir/f'oof_{origin}_r01_lgbm.csv.gz'
        if not op.exists(): continue
        oof=pd.read_csv(op); oof=oof.loc[oof.h.between(1,7)].reset_index(drop=True); oof['source_date']=pd.to_datetime(oof.source_date).dt.to_period('M')
        cutoff=pd.Period(origin,'M'); field,u32,meta=_basis(train,cutoff,32); cells=list(field.columns); lookup={tuple(c):i for i,c in enumerate(cells)}
        prefix=train.loc[pd.to_datetime(train.time).dt.to_period('M')<cutoff].copy(); prefix['period']=pd.to_datetime(prefix.time).dt.to_period('M')
        means=field.mean(axis=0).to_numpy(float); centered=field.sub(means,axis=1).fillna(0.0).to_numpy(float)
        periods=list(field.index); pidx={p:i for i,p in enumerate(periods)}
        # Legal monthly factor transitions plus contemporaneous hydrology projections.
        hydro_month=prefix.groupby('period')[list(HYDRO)].mean().reindex(periods).fillna(0.0).to_numpy(float)
        hx=np.column_stack([hydro_month[:,j] for j in range(hydro_month.shape[1])])
        for rank in RANKS:
            u=u32[:,:min(rank,u32.shape[1])]; z=centered@u; hp=np.nan_to_num(hx,nan=0.0)@np.zeros((len(HYDRO),u.shape[1]))
            feats=[]; ys=[]
            for i in range(len(periods)-1):
                feats.append(np.r_[z[i],z[i+1]-z[i],hydro_month[i],hydro_month[i+1]])
                ys.append(z[i+1]-z[i])
            if not ys: continue
            X=np.asarray(feats); Y=np.asarray(ys); model=Ridge(alpha=10.0).fit(X,Y)
            preds=[]; truth=[]
            for _,row in oof.iterrows():
                anchor=pd.Period(pd.to_datetime(row.anchor_date),'M'); src=row.source_date; ai=max([i for i,p in enumerate(periods) if p<=anchor],default=-1)
                si=max([i for i,p in enumerate(periods) if p<=src],default=ai)
                if ai<0: preds.append(row.prediction); truth.append(row.truth); continue
                si=min(si,len(periods)-1); feat=np.r_[z[ai],np.zeros(u.shape[1]),hydro_month[ai],hydro_month[si]]
                dz=model.predict(feat.reshape(1,-1))[0]; loc=lookup.get((float(row.lat),float(row.lon)))
                preds.append(float(row.anchor_tws + (u[loc]@dz if loc is not None else row.prediction-row.anchor_tws))); truth.append(row.truth)
            pred=np.asarray(preds); y=np.asarray(truth); results.append({'origin':origin,'rank':rank,'rows':len(y),'raw_rmse':float(np.sqrt(np.mean((pred-y)**2))),'mae':float(np.mean(np.abs(pred-y))),'bias':float(np.mean(pred-y)),'r01_rmse':float(np.sqrt(np.mean((oof.prediction.to_numpy()-y)**2))),'ridge_alpha':10.0,'training_months':len(periods),'label':'B2_DEPLOYABLE_SCREEN'})
            pd.DataFrame({'sample_id':oof.sample_id,'source_date':oof.source_date.astype(str),'h':oof.h,'truth':y,'prediction':pred,'r01_prediction':oof.prediction,'origin':origin,'rank':rank}).to_csv(a.output_dir/f'oof_{origin}_b2_rank{rank}.csv.gz',index=False,compression='gzip')
    pd.DataFrame(results).to_csv(a.output_dir/'metrics.csv',index=False); manifest={'status':'completed','stage':'B2','label':'B2_DEPLOYABLE_SCREEN','commit':head,'origins':origins,'ranks':RANKS,'no_test_predictions':True,'elapsed_seconds':time.perf_counter()-started}; (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps({'status':'completed','rows':len(results),'output_dir':str(a.output_dir)}))
if __name__=='__main__': main()
