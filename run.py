#!/usr/bin/env python3
"""Small, non-destructive entry points for the research artifact."""
from __future__ import annotations
import argparse
import gzip
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ.setdefault(name,'1')


def unpack():
    count=0
    for src in (ROOT/'benchmark').rglob('trials.csv.gz'):
        dst=src.with_suffix('')
        if not dst.exists():
            with gzip.open(src,'rb') as inp,dst.open('wb') as out:
                import shutil
                shutil.copyfileobj(inp,out)
            count+=1
    print(f'Restored {count} archived CSV files. Compressed originals unchanged.')


def summary():
    import pandas as pd
    src=ROOT/'benchmark/replication/trials.csv.gz'
    if not src.exists():src=src.with_suffix('')
    d=pd.read_csv(src)
    final=d[d['round']==3]
    table=final.groupby('policy').agg(n=('correct','size'),full_budget_accuracy=('correct','mean'))
    stopped=d[d.stopped_here==1]
    for p in table.index:
        s=stopped[stopped.policy==p]
        table.loc[p,'mean_additional_runs']=s['round'].mean()
        table.loc[p,'unresolved_fraction']=1-s.confident.mean()
    out=ROOT/'outputs/summary';out.mkdir(parents=True,exist_ok=True)
    table.to_csv(out/'recomputed_summary.csv')
    print(table.to_string())
    return table


def demo(order: int,seed: int,sigma: float):
    import numpy as np
    sys.path.insert(0,str(ROOT/'benchmark'))
    from experiment import Bank,obs,truth_draw,dopt_scores,risk_scores
    from models import INITIAL,MENUS
    case='enzyme';bank=Bank(case,order);truth,support=truth_draw(case,seed)
    initial=obs(case,truth,-1,sigma,seed,-1)
    runs={}
    for policy in ('D_OPT','STRUCT_RISK'):
        history=[(-1,initial)];records=[]
        for step in range(4):
            w=bank.posterior(history,sigma);pm,m,coef=bank.summarize(w)
            record=dict(round=step,posterior=pm.tolist(),selected_support=m,coefficients=coef.tolist(),
                        observations_so_far=len(history),maximum_structure_probability=float(pm.max()))
            if pm.max()>=.95 or step==3:
                record['status']='posterior_threshold_reached' if pm.max()>=.95 else 'unresolved_at_budget'
                records.append(record);break
            if policy=='D_OPT':scores=dopt_scores(bank,w,history,sigma)
            else:scores,_=risk_scores(bank,w,sigma,power=5,seed=413+step)
            action=int(np.argmax(scores))
            record.update(next_action=action,next_setting=list(MENUS[case][action]),scores=scores.tolist())
            records.append(record)
            new=obs(case,truth,action,sigma,seed,step)
            history.append((action,new))
        runs[policy]=dict(history=records,initial_setting=list(INITIAL[case]),final_coefficients=records[-1]['coefficients'],
                          actual_additional_experiments=len(history)-1)
        print(policy,'added runs:',len(history)-1,'final structure posterior:',round(records[-1]['maximum_structure_probability'],6),
              'coefficients:',records[-1]['coefficients'])
    out=ROOT/'outputs/demo';out.mkdir(parents=True,exist_ok=True)
    result=dict(case=case,seed=seed,sigma=sigma,order=order,policies=runs,
                truth_for_evaluation_only=truth.tolist(),true_support_for_evaluation_only=support)
    (out/'campaign.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['"""Explicit fitted equations from the latest demo; synthetic data only."""','']
    for p,v in runs.items():
        a,b=v['final_coefficients']
        lines.extend([f'def {p.lower()}_rhs(x):',f'    return -{a!r} * x - {b!r} * x / (0.4 + x)',''])
    (out/'identified_equations.py').write_text('\n'.join(lines))
    print('Saved:',out)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['check','demo','summary','figures','unpack','full'])
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--order',type=int,default=120)
    ap.add_argument('--seed',type=int,default=64000000)
    ap.add_argument('--sigma',type=float,default=.03)
    args=ap.parse_args()
    if args.command=='unpack':unpack()
    elif args.command=='summary':summary()
    elif args.command=='demo':demo(args.order,args.seed,args.sigma)
    elif args.command=='check':runpy.run_path(str(ROOT/'scripts/check_repository.py'),run_name='__main__')
    elif args.command=='figures':
        runpy.run_path(str(ROOT/'scripts/prepare_visual_data.py'),run_name='__main__')
        runpy.run_path(str(ROOT/'scripts/make_readme_assets.py'),run_name='__main__')
    elif args.command=='full':
        command=[sys.executable,str(ROOT/'benchmark/experiment.py'),'--n','192','--order','120','--power','5',
                 '--seed','61000000','--workers',str(args.workers),'--out',str(ROOT/'outputs/full')]
        subprocess.run(command,cwd=ROOT/'benchmark',check=True)

if __name__=='__main__':main()
