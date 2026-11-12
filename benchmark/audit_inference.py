"""Deterministic quadrature and prospective-risk sampling refinement on fixed data."""
from experiment import *

def history_from(row):
    truth=np.array(json.loads(row.truth));path=json.loads(row.history)
    hist=[(-1,obs(row.case,truth,-1,row.sigma,int(row.seed),-1,row.scenario))]
    for s,a in enumerate(path):hist.append((a,obs(row.case,truth,a,row.sigma,int(row.seed),s,row.scenario)))
    return hist

def selected_rows(data,n=12):
    chosen=[]
    data=data[data.policy.isin(['D_OPT','STRUCT_RISK'])].drop_duplicates(['case','sigma','seed','history'])
    for (case,sigma),d in data.groupby(['case','sigma']):
        # Include random states, least certain states and states closest to the stopping boundary.
        ids=list(d.sample(min(4,len(d)),random_state=17).index)
        ids+=list((d.risk-.05).abs().nsmallest(4).index)
        ids+=list(d.risk.nlargest(4).index)
        chosen.append(d.loc[list(dict.fromkeys(ids))])
    return pd.concat(chosen)

def main():
    df=pd.read_csv(ROOT/'replication'/'trials.csv');sel=selected_rows(df)
    rows=[];tic=time.perf_counter()
    for case in CASES:
        b1=getbank(case,120);b2=getbank(case,240)
        for _,r in sel[sel.case==case].iterrows():
            h=history_from(r);p1,m1,e1=b1.summarize(b1.posterior(h,r.sigma));p2,m2,e2=b2.summarize(b2.posterior(h,r.sigma))
            rows.append(dict(case=case,seed=int(r.seed),sigma=r.sigma,round=int(r['round']),history=r.history,max_p_change=float(np.abs(p1-p2).max()),map_changed=int(m1!=m2),confidence_changed=int((p1.max()>=.95)!=(p2.max()>=.95)),coef_change=float(np.linalg.norm(e1-e2)),p120=json.dumps(p1.tolist()),p240=json.dumps(p2.tolist())))
        print('QUAD',case,round(time.perf_counter()-tic,1),flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'audits'/'quadrature.csv',index=False)
    scores=[]
    for case in CASES:
        sub=df[(df.case==case)&(df.policy=='STRUCT_RISK')&(df['round']<3)&(df.risk>.02)].sort_values('risk',ascending=False)
        sub=pd.concat([sub.head(2),sub.sample(min(2,len(sub)),random_state=73)]).drop_duplicates(['seed','history'])
        bank=getbank(case,120)
        for _,r in sub.iterrows():
            h=history_from(r);w=bank.posterior(h,r.sigma);ss=[];meta=[]
            for power in [5,7,8]:
                a,m=risk_scores(bank,w,r.sigma,power,413+int(r['round']));ss.append(a);meta.append(m)
            a,b,c=ss
            scores.append(dict(case=case,seed=int(r.seed),sigma=r.sigma,round=int(r['round']),action32=int(a.argmax()),action128=int(b.argmax()),action256=int(c.argmax()),changed32_256=int(a.argmax()!=c.argmax()),changed128_256=int(b.argmax()!=c.argmax()),regret32=float(c.max()-c[a.argmax()]),max_score_change=float(abs(a-c).max()),bound_excess32=meta[0]['bound_excess'],bound_excess256=meta[2]['bound_excess']))
        print('MC',case,round(time.perf_counter()-tic,1),flush=True)
    pd.DataFrame(scores).to_csv(ROOT/'audits'/'predictive_mc.csv',index=False)
    print('QUADRATURE',pd.DataFrame(rows).max(numeric_only=True).to_dict(),flush=True)
    print('MC',pd.DataFrame(scores).max(numeric_only=True).to_dict(),flush=True)
if __name__=='__main__':main()
