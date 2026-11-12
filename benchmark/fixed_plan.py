"""Fit a nonadaptive three-acquisition schedule using independent prior simulations.
All 35 multisets (with repetitions) are compared, then their selected ordering is
chosen by intermediate risk. This is Monte Carlo, not globally exact optimization.
No replication task, seed or observation is consulted.
"""
from experiment import *

def plan(case,sigma,order=40,n=96):
    bank=getbank(case,order);idx=(CASES+['reaction3']).index(case)
    rng=np.random.default_rng(43000000+idx*137+int(sigma*10000))
    rr=RANGES[case];mask=rng.random((n,len(rr)))<Q
    th=rng.uniform(rr[:,0],rr[:,1],size=mask.shape)*mask
    noises=rng.normal(size=(4,n,bank.dim))
    y0=response(case,th,INITIAL[case])+sigma*noises[0]
    base=np.array(bank.logprior)[None,:]-.5*cdist(y0/sigma,bank.initial/sigma,'sqeuclidean')
    LL=[]
    for s in range(3):
        LL.append([-.5*cdist((response(case,th,ac)+sigma*noises[s+1])/sigma,bank.pred[a]/sigma,'sqeuclidean') for a,ac in enumerate(bank.menu)])
    def risk(seq):
        lw=base.copy()
        for s,a in enumerate(seq):lw+=LL[s][a]
        group=np.column_stack([logsumexp(lw[:,ii],axis=1) for ii in bank.groups])
        pp=np.exp(group-logsumexp(group,axis=1,keepdims=True))
        return float(np.mean(1-pp.max(axis=1)))
    seqs=list(itertools.combinations_with_replacement(range(len(bank.menu)),3))
    scores=np.array([risk(s) for s in seqs]);best=seqs[int(scores.argmin())]
    perms=sorted(set(itertools.permutations(best)))
    riskprefix=[risk(s[:1])+risk(s[:2]) for s in perms];chosen=perms[int(np.argmin(riskprefix))]
    return dict(sequence=list(chosen),prior_risk=float(scores.min()),n_prior_truths=n,order=order,seed=43000000+idx*137+int(sigma*10000),all_schedules=[dict(sequence=list(s),risk=float(r)) for s,r in zip(seqs,scores)])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cases',default=','.join(CASES));ap.add_argument('--order',type=int,default=40);ap.add_argument('--n',type=int,default=96);args=ap.parse_args()
    fn=ROOT/'fixed_plans.json';results=json.loads(fn.read_text()) if fn.exists() else {};tic=time.perf_counter()
    for c in args.cases.split(','):
        for s in [.03,.10]:
            results[f'{c}/{s}']=plan(c,s,args.order,args.n)
            fn.write_text(json.dumps(results,indent=2));print(c,s,results[f'{c}/{s}']['sequence'],round(time.perf_counter()-tic,1),flush=True)
