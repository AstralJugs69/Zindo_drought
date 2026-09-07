"""Causal B1 spatial projection of frozen R01 deltas (Kaggle only)."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scripts.run_spatial_phase_a import _basis

ORIGINS=("2007-09","2009-01","2014-12"); RANKS=(8,16)

def sha(path: Path)->str:
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',type=Path,required=True); ap.add_argument('--run-dir',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--expected-commit',required=True); a=ap.parse_args()
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if head!=a.expected_commit: raise RuntimeError((head,a.expected_commit))
    a.output_dir.mkdir(parents=True,exist_ok=False); started=time.perf_counter()
    train=pd.read_csv(a.data_dir/'Train.csv',usecols=['time','lat','lon','TWS_t'])
    rows=[]; manifests=[]
    for origin in ORIGINS:
        path=a.run_dir/f'oof_{origin}_r01_lgbm.csv.gz'
        if not path.exists(): continue
        oof=pd.read_csv(path); oof=oof.loc[oof.h.between(1,7)].copy().reset_index(drop=True); oof['source_date']=pd.to_datetime(oof.source_date).dt.strftime('%Y-%m-%d')
        field,u32,meta=_basis(train,pd.Period(origin,'M'),32); lookup={tuple(c):i for i,c in enumerate(field.columns)}
        for rank in RANKS:
            u=u32[:,:min(rank,u32.shape[1])]; all_pred=oof.prediction.to_numpy(float).copy(); supported=0
            for date,day in oof.groupby('source_date',sort=True):
                idx=day.index.to_numpy(); loc=[lookup.get((float(x),float(y))) for x,y in zip(day.lat,day.lon)]; keep=np.array([z is not None for z in loc])
                if not keep.any(): continue
                pos=np.flatnonzero(keep); design=u[np.asarray([loc[j] for j in pos],dtype=int),:]; delta=(day.prediction.to_numpy(float)-day.anchor_tws.to_numpy(float))[keep]
                coef=np.linalg.lstsq(design,delta,rcond=None)[0]; all_pred[idx[pos]]=day.anchor_tws.to_numpy(float)[keep]+design@coef; supported+=int(keep.sum())
            y=oof.truth.to_numpy(float); err=all_pred-y; rows.append({'origin':origin,'rank':rank,'rows':int(len(oof)),'supported_rows':supported,'coverage':supported/max(len(oof),1),'raw_rmse':float(np.sqrt(np.mean(err**2))),'mae':float(np.mean(np.abs(err))),'bias':float(np.mean(err)),'r01_rmse':float(np.sqrt(np.mean((oof.prediction.to_numpy(float)-y)**2))),'label':'B1_DEPLOYABLE_SCREEN'})
            pd.DataFrame({'sample_id':oof.sample_id,'source_date':oof.source_date,'h':oof.h,'lat':oof.lat,'lon':oof.lon,'truth':y,'prediction':all_pred,'r01_prediction':oof.prediction,'origin':origin,'rank':rank}).to_csv(a.output_dir/f'oof_{origin}_b1_rank{rank}.csv.gz',index=False,compression='gzip')
        manifests.append({'origin':origin,'basis_cells':int(field.shape[1]),'basis_rank':int(u32.shape[1]),'basis_train_rows':meta.get('rows',0)})
    pd.DataFrame(rows).to_csv(a.output_dir/'metrics.csv',index=False)
    manifest={'status':'completed','stage':'B1','label':'B1_DEPLOYABLE_SCREEN','commit':head,'origins':ORIGINS,'ranks':RANKS,'basis':manifests,'no_test_predictions':True,'elapsed_seconds':time.perf_counter()-started,'checksums':{p.name:sha(p) for p in a.output_dir.glob('*.csv*')}}
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps({'status':'completed','rows':len(rows),'output_dir':str(a.output_dir)}))
if __name__=='__main__': main()
