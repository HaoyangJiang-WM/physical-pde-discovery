"""Paired diagnostic only: the true AR(1) observation correlation is supplied.
No new selector, no learned covariance, no claim that noisy correlation is solved.
Shows what changes when BOTH selectors receive the correct same likelihood.
"""
import experiment as ex
from experiment import *
from functools import lru_cache
RHO=.7
raw_getbank=ex.getbank
raw_obs=ex.obs
raw_jacobian=ex.jacobian

def whiten(y):
    y=np.asarray(y);shape=y.shape;v=y.reshape(*shape[:-1],len(TIMES),-1);w=v.copy()
    w[...,1:,:]=(v[...,1:,:]-RHO*v[...,:-1,:])/np.sqrt(1-RHO**2)
    return w.reshape(shape)

class WhiteBank(ex.Bank):
    def __init__(self,case,order):
        b=raw_getbank(case,order);self.__dict__.update(b.__dict__)
        self.initial=whiten(b.initial);self.pred=whiten(b.pred)
        # held trajectories remain unwhitened for physical prediction scoring.

@lru_cache(None)
def getwhite(case,order):return WhiteBank(case,order)

def whiteobs(*args,**kwargs):return whiten(raw_obs(*args,**kwargs))
def whitejac(*args,**kwargs):return whiten(raw_jacobian(*args,**kwargs).T).T

def task(t):
    ex.getbank=getwhite;ex.obs=whiteobs;ex.jacobian=whitejac
    return ex.run_one(t)

if __name__=='__main__':
    jobs=[(c,92000000+ci*100000+i,.10,120,5,'correlated',['D_OPT','STRUCT_RISK']) for ci,c in enumerate(['enzyme','forced_damping','reaction']) for i in range(64)]
    out=[];tic=time.perf_counter()
    with ProcessPoolExecutor(3) as pool:
        for i,r in enumerate(pool.map(task,jobs,chunksize=1)):
            out.extend(r)
            if (i+1)%32==0:print('NOISE_AWARE',i+1,len(jobs),round(time.perf_counter()-tic,1),flush=True)
    ex.summarize(out,ROOT/'stress'/'noise_aware')
    (ROOT/'stress'/'noise_aware'/'protocol.json').write_text(json.dumps(dict(paired_with='stress/robustness, correlated subset',known_rho=RHO,covariance='oracle known, not estimated',independent_new_tasks=0,paired_tasks=192),indent=2))
