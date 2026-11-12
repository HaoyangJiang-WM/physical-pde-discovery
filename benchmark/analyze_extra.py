"""Regenerate the paired noise-model and three-term summaries."""
from pathlib import Path
import pandas as pd
from analyze import paired_ci
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    raw=pd.read_csv(ROOT/'stress'/'robustness'/'trials.csv')
    revised=pd.read_csv(ROOT/'stress'/'noise_aware'/'trials.csv')
    out=[]
    for name,d in [('wrong_iid',raw[raw.scenario=='correlated']),('known_AR1',revised)]:
        f=d[d['round']==3];s=d[d.stopped_here==1]
        a=f.groupby('policy').agg(n=('correct','size'),full_accuracy=('correct','mean'))
        b=s.groupby('policy').apply(lambda x:pd.Series(dict(mean_runs=x['round'].mean(),confident_fraction=x.confident.mean(),accuracy_if_confident=x.loc[x.confident==1,'correct'].mean(),forced_MAP_stopped=x.correct.mean())),include_groups=False)
        z=a.join(b);z['inference']=name;out.append(z.reset_index())
    pd.concat(out).to_csv(ROOT/'noise_model_comparison.csv',index=False)
    d=pd.read_csv(ROOT/'stress'/'reaction3'/'trials.csv');d=d[d['round']==3]
    out=[]
    for sigma in ['ALL',.03,.10]:
        x=d if sigma=='ALL' else d[d.sigma==sigma]
        w=x.pivot(index=['seed','sigma'],columns='policy',values='correct')
        for b in ['D_OPT','FIXED','RANDOM']:
            ci=paired_ci((w.STRUCT_RISK-w[b]).to_numpy())
            out.append(dict(sigma=sigma,baseline=b,n=len(w),difference_pp=100*ci[0],low_pp=100*ci[1],high_pp=100*ci[2]))
    pd.DataFrame(out).to_csv(ROOT/'three_term_paired.csv',index=False)
    print(pd.DataFrame(out).to_string(index=False))
