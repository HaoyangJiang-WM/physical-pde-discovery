"""Export deterministic illustrative cases; not additional performance evidence."""
from experiment import *
if __name__=='__main__':
    df=pd.read_csv(ROOT/'replication'/'trials.csv');out=ROOT/'examples';out.mkdir(exist_ok=True)
    picked=[]
    for case in ['enzyme','duffing','forced_damping','allen_cahn']:
        d=df[(df.case==case)&(df.policy=='STRUCT_RISK')]
        init=d[d['round']==0].set_index('seed');end=d[d['round']==3].set_index('seed')
        seeds=sorted(set(init.index[(init.correct==0)&(init.risk>.05)])&set(end.index[end.correct==1]))
        if not seeds:continue
        seed=seeds[0];hist=d[d.seed==seed].sort_values('round');r0=hist.iloc[0]
        # Show first added acquisition after which the support is correct, with all previous fits retained in data.
        r1=hist[(hist['round']>0)&(hist.correct==1)].iloc[0];stage=int(r1['round'])-1
        path=json.loads(r1.history);ai=path[-1];truth=np.array(json.loads(r0.truth));setting=MENUS[case][ai]
        t=np.linspace(0,T,241)
        fits=np.array([json.loads(r.estimate) for _,r in hist[hist['round']<=r1['round']].iterrows()])
        curves=np.array([response(case,th,setting,times=t,substeps=4)[0] for th in fits])
        yy=obs(case,truth,ai,float(r0.sigma),int(seed),stage,'matched')
        arrays=dict(times=TIMES,dense_times=t,observed=yy,fits=fits,curves=curves,truth=truth,setting=np.array(setting))
        np.savez_compressed(out/f'{case}.npz',**arrays)
        record=dict(case=case,seed=int(seed),sigma=float(r0.sigma),rule='lowest seed initially wrong and uncertain, ultimately correct; earliest correct update',chosen_round=int(r1['round']),setting=setting,initial_theta=json.loads(r0.estimate),new_theta=json.loads(r1.estimate),true_theta=truth.tolist(),old_probabilities=json.loads(r0.probabilities),new_probabilities=json.loads(r1.probabilities),equation=EQUATIONS[case],fixed_T=T)
        (out/f'{case}.json').write_text(json.dumps(record,indent=2));hist.to_csv(out/f'{case}_history.csv',index=False);picked.append(record)
    (out/'index.json').write_text(json.dumps(picked,indent=2));print(json.dumps(picked,indent=2))
