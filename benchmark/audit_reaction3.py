from audit_inference import *
if __name__=='__main__':
    import time
    while not (ROOT/'stress'/'reaction3'/'runtime.json').exists():time.sleep(2)
    df=pd.read_csv(ROOT/'stress'/'reaction3'/'trials.csv');sel=selected_rows(df)
    b1=getbank('reaction3',48);b2=getbank('reaction3',64);rows=[]
    for _,r in sel.iterrows():
        h=history_from(r);p1,m1,t1=b1.summarize(b1.posterior(h,r.sigma));p2,m2,t2=b2.summarize(b2.posterior(h,r.sigma))
        rows.append(dict(seed=int(r.seed),sigma=r.sigma,round=int(r['round']),max_p_change=float(abs(p1-p2).max()),map_changed=int(m1!=m2),confidence_changed=int((p1.max()>=.95)!=(p2.max()>=.95)),p48=json.dumps(p1.tolist()),p64=json.dumps(p2.tolist()),history=r.history))
    pd.DataFrame(rows).to_csv(ROOT/'audits'/'reaction3_quadrature.csv',index=False)
    print(pd.DataFrame(rows).max(numeric_only=True).to_dict(),flush=True)
