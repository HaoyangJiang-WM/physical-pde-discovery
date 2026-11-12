"""Paired fixed-duration equation-discovery benchmark.
A: local full-library D-optimality. B: posterior structural decision risk.
Identical spike-and-slab inference, observations, priors and stopping for both.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[k]='1'
import argparse, json, time, itertools, hashlib
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.special import logsumexp,ndtri
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from numpy.polynomial.legendre import leggauss
from models import *

ROOT=Path(__file__).resolve().parent
POLICIES=['RANDOM','FIXED','D_OPT','STRUCT_RISK']
CONFIDENCE=.95
MAX_STEPS=3
PRUNE=1e-10

class Bank:
    def __init__(self,case,order=120):
        self.case=case;self.order=order;self.menu=MENUS[case];self.p=len(RANGES[case]);self.nmod=2**self.p
        tag=ROOT/'cache'/f'{case}_{order}'
        if not (tag/'ready.json').exists():
            tag.mkdir(exist_ok=True,parents=True)
            z,gw=leggauss(order);weights=np.r_[1-Q,Q*gw/2]
            ind=np.indices((order+1,)*self.p).reshape(self.p,-1).T
            theta=np.column_stack([np.r_[0,(z+1)*(hi-lo)/2+lo][ind[:,j]] for j,(lo,hi) in enumerate(RANGES[case])])
            labels=np.sum((ind>0)*(2**np.arange(self.p)),axis=1).astype(np.int16)
            weight=np.prod(weights[ind],axis=1)
            np.save(tag/'theta.npy',theta);np.save(tag/'labels.npy',labels);np.save(tag/'logprior.npy',np.log(weight))
            init=response(case,theta,INITIAL[case]);np.save(tag/'initial.npy',init)
            pred=np.stack([response(case,theta,ac) for ac in self.menu]);np.save(tag/'pred.npy',pred)
            held=np.stack([response(case,theta,ac) for ac in HELDOUT[case]])
            np.save(tag/'held.npy',held)
            if not (np.isfinite(pred).all() and np.isfinite(init).all() and np.isfinite(held).all()):raise RuntimeError('non-finite model bank')
            (tag/'ready.json').write_text(json.dumps(dict(case=case,order=order,n_nodes=len(theta),shape=list(pred.shape),equation=EQUATIONS[case])))
        for name in ['theta','labels','logprior','initial','pred','held']:
            setattr(self,name,np.load(tag/f'{name}.npy',mmap_mode='r'))
        self.dim=self.initial.shape[1]
        self.groups=[np.flatnonzero(self.labels==s) for s in range(self.nmod)]
    def posterior(self,history,sigma):
        logw=np.array(self.logprior)
        for e,y in history:
            mu=self.initial if e<0 else self.pred[e]
            logw-=.5*np.sum((mu-y)**2,axis=1)/sigma**2
        return np.exp(logw-logsumexp(logw))
    def summarize(self,w):
        pm=np.bincount(self.labels,weights=w,minlength=self.nmod);m=int(np.argmax(pm))
        ii=self.groups[m];est=(w[ii]/pm[m])@self.theta[ii]
        return pm,m,est

@lru_cache(None)
def getbank(case,order):return Bank(case,order)

@lru_cache(maxsize=128)
def draws(power,seed,dim):
    q=qmc.Sobol(dim+1,scramble=True,seed=seed).random_base2(power)
    return q[:,0],ndtri(np.clip(q[:,1:],1e-12,1-1e-12))

def risk_scores(bank,w,sigma,power=5,seed=413):
    pm,star,_=bank.summarize(w);upper=1-pm.max()
    if upper<1e-10:return np.zeros(len(bank.menu)),dict(raw_max=0.,bound_excess=0.,pruned=0.,retained=len(w))
    order=np.argsort(w)[::-1];n=min(len(w),np.searchsorted(np.cumsum(w[order]),1-PRUNE)+1)
    ids=order[:n];ww=w[ids];pruned=max(0.,1-ww.sum());ww=ww/ww.sum();lp=np.log(ww)
    gen=[];mass=[];noise=[]
    for s,ii in enumerate(bank.groups):
        if pm[s]<1e-13:continue
        u,z=draws(power,seed+37*s,bank.dim)
        jj=np.minimum(np.searchsorted(np.cumsum(w[ii]/pm[s]),u),len(ii)-1)
        gen.extend(ii[jj]);mass.extend(np.full(len(jj),pm[s]/len(jj)));noise.extend(z)
    gen=np.asarray(gen);mass=np.asarray(mass);mass/=mass.sum();noise=np.asarray(noise)
    labels=bank.labels[ids];raw=[]
    for a in range(len(bank.menu)):
        mu=bank.pred[a,ids]/sigma;y=bank.pred[a,gen]/sigma+noise
        logpost=-.5*cdist(y,mu,'sqeuclidean')+lp[None,:]
        lg=np.column_stack([logsumexp(logpost[:,labels==s],axis=1) if np.any(labels==s) else np.full(len(y),-np.inf) for s in range(bank.nmod)])
        pp=np.exp(lg-logsumexp(lg,axis=1,keepdims=True))
        raw.append(float(mass@(pp.max(axis=1)-pp[:,star])))
    raw=np.asarray(raw)
    # The control variate yields nonnegative samples; finite integration can exceed the upper bound.
    # Preserve the raw scores in logs and audit integration refinement separately.
    return np.clip(raw,0,upper),dict(raw_scores=json.dumps(raw.tolist()),raw_max=float(raw.max()),bound_excess=float(max(0,raw.max()-upper)),pruned=pruned,retained=n)

def dopt_scores(bank,w,history,sigma):
    _,_,theta=bank.summarize(w);lo,hi=RANGES[bank.case].T
    var=Q*(lo*lo+lo*hi+hi*hi)/3-Q**2*((lo+hi)/2)**2
    G=np.diag(1/var)
    for e,_ in history:
        J=jacobian(bank.case,theta,INITIAL[bank.case] if e<0 else bank.menu[e])/sigma
        G+=J.T@J
    start=np.linalg.slogdet(G)[1];out=[]
    for ac in bank.menu:
        J=jacobian(bank.case,theta,ac)/sigma
        out.append(float(np.linalg.slogdet(G+J.T@J)[1]-start))
    return np.asarray(out)

def truth_draw(case,seed,scenario='matched'):
    rng=np.random.default_rng(np.random.SeedSequence([seed,67531]));rr=RANGES[case]
    support=rng.random(len(rr))<Q
    th=rng.uniform(rr[:,0],rr[:,1])*support
    if scenario=='near_zero':support[-1]=True;th[-1]=rng.uniform(.001,.035)*rr[-1,1]
    if scenario=='dense_prior':
        support=rng.random(len(rr))<.8;th=rng.uniform(rr[:,0],rr[:,1])*support
    return th,int(np.sum(support*2**np.arange(len(rr))))

def clean_response(case,truth,setting,scenario='matched'):
    if scenario=='missing_cubic' and case=='reaction':
        return response('reaction3',np.r_[truth,.25],setting)[0]
    if scenario=='fine_grid' and case=='allen_cahn':
        return response(case,truth,setting,substeps=32,nx=64)[0]
    return response(case,truth,setting)[0]

def obs(case,truth,action,sigma,seed,stage,scenario='matched'):
    setting=INITIAL[case] if action<0 else MENUS[case][action]
    mu=clean_response(case,truth,setting,scenario)
    rng=np.random.default_rng(np.random.SeedSequence([seed,stage+1,action+1,331923]))
    if case=='diffusion_pde':
        k=setting[0];x=np.arange(16)*2*np.pi/16;v=np.cos(k*x)
        noise=rng.normal(size=(len(TIMES),16))@v/np.sqrt(8)
    elif case=='advection_diffusion':
        k=setting[0];x=np.arange(32)*2*np.pi/32;v=np.column_stack([np.cos(k*x),np.sin(k*x)])
        noise=(rng.normal(size=(len(TIMES),32))@v/4).ravel()
    elif case=='oscillator':
        noise=np.sum(rng.normal(size=(len(TIMES),2))*np.column_stack([np.cos(TIMES),np.sin(TIMES)]),axis=1)
    else:noise=rng.normal(size=mu.shape)
    if scenario=='correlated':
        noise=noise.reshape(len(TIMES),-1)
        for t in range(1,len(TIMES)):noise[t]=.7*noise[t-1]+np.sqrt(1-.7**2)*noise[t]
        noise=noise.ravel()
    if scenario=='noise_double':noise*=2
    return mu+sigma*noise

def fixed_sequence(case,sigma):
    f=ROOT/'fixed_plans.json'
    if f.exists():
        d=json.loads(f.read_text())
        if f'{case}/{sigma}' in d:return d[f'{case}/{sigma}']['sequence']
    return [0,4,2]

def run_one(task):
    case,seed,sigma,order,power,scenario,policies=task
    bank=getbank(case,order);truth,true_mask=truth_draw(case,seed,scenario)
    initial=obs(case,truth,-1,sigma,seed,-1,scenario)
    held_truth=np.stack([clean_response(case,truth,ac,scenario) for ac in HELDOUT[case]])
    fixed=fixed_sequence(case,sigma);cache={};rows=[]
    for policy in policies:
        path=[];history=[(-1,initial)];first_stop=None
        for step in range(MAX_STEPS+1):
            key=tuple(path)
            if key not in cache:
                w=bank.posterior(history,sigma);pm,m,est=bank.summarize(w)
                ii=np.flatnonzero(w>1e-14);w2=w[ii]/w[ii].sum()
                pred=np.array([w2@h[ii] for h in bank.held])
                cache[key]=dict(w=w,pm=pm,m=m,estimate=est,forecast_rmse=float(np.sqrt(np.mean((pred-held_truth)**2))))
            state=cache[key];pm=state['pm'];risk=float(1-pm.max());confident=risk<=1-CONFIDENCE
            stop_reason='confident' if confident else 'budget' if step==MAX_STEPS else ''
            if first_stop is None and stop_reason:first_stop=step
            row=dict(case=case,seed=seed,sigma=sigma,order=order,power=power,scenario=scenario,policy=policy,round=step,
                true_mask=true_mask,pred_mask=state['m'],correct=int(state['m']==true_mask and scenario!='missing_cubic'),
                in_library_support_correct=int(state['m']==true_mask),model_in_library=int(scenario!='missing_cubic'),
                risk=risk,p_true=float(pm[true_mask]),probabilities=json.dumps(pm.tolist()),
                brier=float(np.sum((pm-np.eye(bank.nmod)[true_mask])**2)),
                truth=json.dumps(truth.tolist()),estimate=json.dumps(state['estimate'].tolist()),
                coef_scaled_rmse=float(np.sqrt(np.mean(((state['estimate']-truth)/RANGES[case][:,1])**2))),
                history=json.dumps(path),stop_reason=stop_reason,first_stop=first_stop if first_stop is not None else -1,
                stopped_here=int(first_stop==step),confident=int(confident),forecast_rmse=state['forecast_rmse'])
            if step<MAX_STEPS:
                tic=time.perf_counter();meta={}
                if policy=='RANDOM':ai=int(np.random.default_rng(np.random.SeedSequence([seed,step,941])).integers(len(bank.menu)));score=np.zeros(len(bank.menu))
                elif policy=='FIXED':ai=int(fixed[step]);score=np.zeros(len(bank.menu))
                elif policy=='REPEAT':ai=min(range(len(bank.menu)),key=lambda a:np.linalg.norm(np.asarray(bank.menu[a])-INITIAL[case]));score=np.zeros(len(bank.menu))
                elif policy=='D_OPT':score=dopt_scores(bank,state['w'],history,sigma);ai=int(np.argmax(score))
                elif policy=='STRUCT_RISK':
                    score,meta=risk_scores(bank,state['w'],sigma,power,seed=413+step)
                    ai=int(np.argmax(score))
                else:raise ValueError(policy)
                if scenario=='restricted':ai=min(range(len(bank.menu)),key=lambda a:np.linalg.norm(np.asarray(bank.menu[a])-INITIAL[case]))
                row.update(action=ai,setting=json.dumps(bank.menu[ai]),selected_score=float(score[ai]),scores=json.dumps(score.tolist()),design_seconds=time.perf_counter()-tic,**meta)
                new=obs(case,truth,ai,sigma,seed,step,scenario);path.append(ai);history.append((ai,new))
            rows.append(row)
    return rows

def summarize(rows,out):
    out=Path(out);out.mkdir(exist_ok=True,parents=True);df=pd.DataFrame(rows);df.to_csv(out/'trials.csv',index=False)
    df.groupby(['case','sigma','scenario','policy','round']).agg(n=('correct','size'),accuracy=('correct','mean'),risk=('risk','mean'),brier=('brier','mean'),forecast_rmse=('forecast_rmse','median'),coef_scaled_rmse=('coef_scaled_rmse','median'),confident=('confident','mean')).reset_index().to_csv(out/'summary.csv',index=False)
    stop=df[df.stopped_here==1]
    stop.groupby(['case','sigma','scenario','policy']).apply(lambda d:pd.Series(dict(n=len(d),mean_runs=d['round'].mean(),forced_map_accuracy=d.correct.mean(),confident_fraction=d.confident.mean(),confident_accuracy=d.loc[d.confident==1,'correct'].mean(),unresolved=(d.confident==0).mean(),false_confident=((d.confident==1)&(d.correct==0)).mean(),risk=d.risk.mean(),forecast_rmse=d.forecast_rmse.median())),include_groups=False).reset_index().to_csv(out/'stopping.csv',index=False)
    return df

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cases',default=','.join(CASES));ap.add_argument('--n',type=int,default=128);ap.add_argument('--order',type=int,default=120);ap.add_argument('--power',type=int,default=5);ap.add_argument('--sigmas',default='.03,.10');ap.add_argument('--seed',type=int,default=61000000);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--policies',default=','.join(POLICIES));ap.add_argument('--scenario',default='matched');ap.add_argument('--out',default='replication');ap.add_argument('--build-only',action='store_true');args=ap.parse_args()
    out=ROOT/args.out;out.mkdir(exist_ok=True,parents=True);cases=args.cases.split(',')
    (out/'protocol.json').write_text(json.dumps(vars(args)|dict(T=T,times=TIMES.tolist(),Q=Q,max_steps=MAX_STEPS,confidence=CONFIDENCE,stop='confidence >= .95 or budget only; no one-step-value stop',source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['models.py','experiment.py']}),indent=2))
    tic=time.perf_counter()
    for case in cases:
        b=getbank(case,args.order);print('BANK',case,len(b.theta),b.pred.shape,'elapsed',round(time.perf_counter()-tic,1),flush=True)
    if args.build_only:return
    tasks=[(c,args.seed+ci*1000000+si*100000+i,s,args.order,args.power,args.scenario,args.policies.split(',')) for ci,c in enumerate(cases) for si,s in enumerate(map(float,args.sigmas.split(','))) for i in range(args.n)]
    rows=[]
    if args.workers>1:
        with ProcessPoolExecutor(args.workers) as pool:
            for j,res in enumerate(pool.map(run_one,tasks,chunksize=1)):
                rows.extend(res)
                if (j+1)%32==0:print('DONE',j+1,'/',len(tasks),'seconds',round(time.perf_counter()-tic,1),flush=True)
    else:
        for j,task in enumerate(tasks):
            rows.extend(run_one(task))
            if (j+1)%8==0:print('DONE',j+1,'/',len(tasks),'seconds',round(time.perf_counter()-tic,1),flush=True)
    summarize(rows,out)
    (out/'runtime.json').write_text(json.dumps(dict(seconds=time.perf_counter()-tic,unique_systems=len(tasks),policies=args.policies.split(',')),indent=2))
    print('COMPLETE',out,'seconds',round(time.perf_counter()-tic,1),flush=True)
if __name__=='__main__':main()
