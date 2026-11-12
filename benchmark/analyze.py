"""Paired, system-level analysis. No strategies share true parameters or future data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parent

def paired_ci(x,seed=431,n_boot=5000):
    x=np.asarray(x,dtype=float);rng=np.random.default_rng(seed)
    z=np.array([rng.choice(x,size=(min(250,n_boot-i),len(x)),replace=True).mean(axis=1) for i in range(0,n_boot,250)]).ravel()
    return float(x.mean()),*np.quantile(z,[.025,.975]).tolist()

def main():
    d=pd.read_csv(ROOT/'replication'/'trials.csv');final=d[d['round']==3];st=d[d.stopped_here==1]
    agg=final.groupby(['case','policy']).agg(n=('correct','size'),accuracy=('correct','mean'),brier=('brier','mean'),forecast_rmse=('forecast_rmse','median'),coef_scaled_rmse=('coef_scaled_rmse','median')).reset_index()
    agg.to_csv(ROOT/'case_results.csv',index=False)
    initial=d[d['round']==0].drop_duplicates(['case','seed','sigma'])
    uncertainty=initial[['case','seed','sigma','confident','correct']].rename(columns={'confident':'initial_confident','correct':'initial_correct'})
    final=final.merge(uncertainty,on=['case','seed','sigma']);st=st.merge(uncertainty,on=['case','seed','sigma'])
    tables=[]
    for subset,label in [(final,'all'),(final[final.true_mask>0],'nonempty'),(final[final.initial_confident==0],'initially_uncertain')]:
        tab=subset.groupby('policy').agg(n=('correct','size'),accuracy=('correct','mean'),brier=('brier','mean'),forecast_rmse=('forecast_rmse','median'),coef_scaled_rmse=('coef_scaled_rmse','median')).reset_index();tab['subset']=label;tables.append(tab)
    pd.concat(tables).to_csv(ROOT/'overall.csv',index=False)
    stops=st.groupby(['case','policy']).apply(lambda x:pd.Series(dict(n=len(x),mean_runs=x['round'].mean(),forced_map_accuracy=x.correct.mean(),confident_fraction=x.confident.mean(),confident_accuracy=x.loc[x.confident==1,'correct'].mean(),unresolved=(x.confident==0).mean(),risk=x.risk.mean())),include_groups=False).reset_index();stops.to_csv(ROOT/'case_stopping.csv',index=False)
    def sm(x):return pd.Series(dict(n=len(x),mean_runs=x['round'].mean(),forced_map_accuracy=x.correct.mean(),confident_fraction=x.confident.mean(),confident_accuracy=x.loc[x.confident==1,'correct'].mean(),unresolved=(x.confident==0).mean(),false_confident=((x.confident==1)&(x.correct==0)).mean(),mean_reported_error=x.risk.mean()))
    ss=st.groupby('policy').apply(sm,include_groups=False).reset_index();ss.to_csv(ROOT/'overall_stopping.csv',index=False)
    st[st.initial_confident==0].groupby('policy').apply(sm,include_groups=False).reset_index().to_csv(ROOT/'uncertain_stopping.csv',index=False)
    comps=[]
    for case in ['ALL']+sorted(final.case.unique()):
        x=final if case=='ALL' else final[final.case==case]
        p=x.pivot(index=['case','seed','sigma'],columns='policy',values='correct')
        for baseline in ['D_OPT','FIXED','RANDOM']:
            dif=p.STRUCT_RISK-p[baseline];ci=paired_ci(dif.to_numpy(),431)
            win=int((dif>0).sum());loss=int((dif<0).sum())
            prob=binomtest(win,win+loss,.5).pvalue if win+loss else 1.
            comps.append(dict(case=case,baseline=baseline,n=len(dif),diff_pp=ci[0]*100,low_pp=ci[1]*100,high_pp=ci[2]*100,wins=win,losses=loss,p_raw=prob))
    cmp=pd.DataFrame(comps)
    cc=(cmp.case!='ALL')&(cmp.baseline=='D_OPT');ids=cmp[cc].sort_values('p_raw').index;adj=np.maximum.accumulate([min(1,float(cmp.loc[j,'p_raw'])*(len(ids)-i)) for i,j in enumerate(ids)])
    cmp.loc[ids,'p_holm_11']=adj;cmp.to_csv(ROOT/'paired_comparisons.csv',index=False)
    d.groupby(['case','sigma','policy','round']).correct.mean().reset_index().to_csv(ROOT/'learning_curves.csv',index=False)
    info=initial.groupby('case').agg(initial_accuracy=('correct','mean'),initial_confident=('confident','mean'),n=('correct','size'),empty_fraction=('true_mask',lambda x:(x==0).mean())).reset_index();info.to_csv(ROOT/'initial_difficulty.csv',index=False)
    # Calibration is a model-based posterior diagnostic, not a frequentist coverage claim.
    cal=d[d.policy.isin(['D_OPT','STRUCT_RISK'])].copy();cal['bin']=pd.cut(1-cal.risk,[0,.5,.7,.9,.95,.99,1.000001],include_lowest=True)
    cal.groupby(['policy','round','bin'],observed=True).agg(n=('correct','size'),correct=('correct','mean'),reported_confidence=('risk',lambda x:1-x.mean())).reset_index().to_csv(ROOT/'posterior_calibration.csv',index=False)
    # Design-time and actual action diversity measured only before the common stopping point.
    active=d[(d['round']<3)&((d.first_stop<0)|(d['round']<d.first_stop))]
    # Correct filtering: rows after a first-stop keep that first stop, so they are excluded.
    active.groupby('policy').design_seconds.agg(['median','mean','count']).to_csv(ROOT/'design_runtime.csv')
    active.groupby(['case','policy','round','action']).size().reset_index(name='n').to_csv(ROOT/'active_action_counts.csv',index=False)
    main_lock=json.loads((ROOT/'REPLICATION_LOCK.json').read_text())
    import hashlib
    integrity={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in main_lock['files'].items()}
    (ROOT/'lock_audit.json').write_text(json.dumps(integrity,indent=2))
    print(agg.pivot(index='case',columns='policy',values='accuracy').mul(100).round(2).to_string())
    print('\nOVERALL\n',pd.concat(tables).to_string(index=False));print('\nSTOP\n',ss.to_string(index=False));print('\nPAIRED\n',cmp.to_string(index=False));print('LOCK',integrity)
if __name__=='__main__':main()
