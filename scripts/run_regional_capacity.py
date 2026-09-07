"""Kaggle-only paired regional-hydrology experiment (no Test predictions)."""
from __future__ import annotations
import argparse, json, time, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_experiment import _attach_delta, _coverage_hash, _json, _preflight, _score
from scripts.run_lgbm_core import DEFAULT_PARAMS
from src.ml_features import SOURCE_CORE_COLUMNS, SOURCE_HYDRO_HISTORY_COLUMNS, build_sampled_training_rows, build_hydro_gap_safe_feature_matrix, horizon_rebalance_weights
from src.observation_simulator import SimulatedFold, build_mask_block_fold, build_template_replay_fold
from src.validation import build_test_mask_template

def regional(frame: pd.DataFrame, width: float) -> pd.DataFrame:
    x = frame.copy(); x["source_date"] = pd.to_datetime(x["time"]).dt.strftime("%Y-%m-%d")
    x["rb_lat"] = np.floor(x.lat / width) * width; x["rb_lon"] = np.floor(x.lon / width) * width
    cols = [c for c in ["SPEI_01_t","SPEI_03_t","SPEI_06_t","SPEI_12_t","SOIL_MOISTURE_t"] if c in x]
    g = x.groupby(["source_date","rb_lat","rb_lon"], sort=False)[cols].transform("mean")
    out = g.to_numpy(float); local = x[cols].to_numpy(float) - out
    return pd.DataFrame(np.c_[out, local, np.isfinite(out).astype(float)], index=frame.index,
        columns=[f"reg{width:g}_{c}" for c in cols] + [f"dev{width:g}_{c}" for c in cols] + [f"reg{width:g}_present_{i}" for i in range(len(cols))])

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--expected-commit',required=True); args=ap.parse_args()
    repo=Path.cwd(); head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if head != args.expected_commit: raise RuntimeError((head,args.expected_commit))
    import lightgbm as lgb
    out=args.output_dir; out.mkdir(parents=True,exist_ok=False); started=time.perf_counter(); train=pd.read_csv(args.data_dir/'Train.csv'); test=pd.read_csv(args.data_dir/'Test.csv'); template=build_test_mask_template(test)
    structural=train[['sample_id','time','lat','lon','TWS_t']]; source=train[SOURCE_HYDRO_HISTORY_COLUMNS]; labels=train[['sample_id','target']]
    params=dict(DEFAULT_PARAMS,num_threads=4,seed=20260907,feature_fraction_seed=20260907,bagging_seed=20260907,data_random_seed=20260907)
    results=[]; region_hash={}
    for origin in ('2007-09','2009-01','2014-12'):
        if origin=='2014-12':
            full=build_mask_block_fold(train,anchor_month=origin,end_month='2015-06',scenario_id='regional_recent'); keep=full.ledger.h.between(1,7).to_numpy(); fold=SimulatedFold(full.spec,full.ledger.loc[keep].reset_index(drop=True),full.labels.loc[keep].reset_index(drop=True),dict(full.exclusions,outside_competition_horizon=int((~keep).sum())))
        else: fold=build_template_replay_fold(train,template,start_month=origin,scenario_id=f'regional_{origin}',family='development')
        sampled=build_sampled_training_rows(structural,source[SOURCE_CORE_COLUMNS],max_target_month=pd.Period(origin,'M')-1); rows=sampled.rows; y=_attach_delta(rows,labels); w=horizon_rebalance_weights(rows.h)
        x0=build_hydro_gap_safe_feature_matrix(rows,source,structural); v0=build_hydro_gap_safe_feature_matrix(fold.ledger,source,structural)
        rtrain=regional(train,5.0); rvalid=regional(train,5.0)
        # sampled/ledger indices are keyed by sample_id; preserve exact pairing.
        rmap=rtrain.assign(sample_id=train.sample_id).set_index('sample_id'); rvmap=rvalid.assign(sample_id=train.sample_id).set_index('sample_id')
        addx=rmap.reindex(rows.sample_id).to_numpy(float); addv=rvmap.reindex(fold.ledger.sample_id).to_numpy(float)
        addx=np.nan_to_num(addx,nan=0.0); addv=np.nan_to_num(addv,nan=0.0); region_hash[origin]=hashlib.sha256(np.ascontiguousarray(addx).tobytes()).hexdigest() if False else str(addx.shape)
        for candidate, xx, vv in [('C0_frozen173',x0,v0),('C1_regional_frozen173',pd.DataFrame(np.c_[x0.to_numpy(),addx]),pd.DataFrame(np.c_[v0.to_numpy(),addv]))]:
            booster=lgb.train(params,lgb.Dataset(xx,label=y,weight=w),num_boost_round=173); pred=fold.ledger.last_observed_TWS.to_numpy(float)+booster.predict(vv); score,_=_score(candidate,fold,'recent_stress' if origin=='2014-12' else 'development',pred); score.update(origin=origin,training_rows=len(rows),capacity='frozen_173',regional_width=5.0); results.append(score); booster.free_dataset()
    _json(out/'metrics.json',{'results':results,'capacity_note':'paired frozen-173 smoke/comparison; inner capacity selection deferred','region_hash':region_hash})
    _json(out/'manifest.json',{'status':'completed','commit':head,'branch':'codex/validation-rebuild','elapsed_seconds':time.perf_counter()-started,'candidates':['C0_frozen173','C1_regional_frozen173'],'no_test_predictions':True,'regional_block':'5-degree contemporaneous means, deviations, presence flags','capacity_selection':'deferred after paired smoke'})
    print(json.dumps({'status':'completed','rows':len(results),'output_dir':str(out)}))
if __name__=='__main__':
    import hashlib
    main()
