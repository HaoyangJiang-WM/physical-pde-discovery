"""Separate stress tests; never pooled with matched-prior validation."""
from experiment import *
if __name__=='__main__':
    groups=[('restricted',['chemostat','diffusion_pde'],64,.10),
            ('correlated',['enzyme','forced_damping','reaction'],64,.10),
            ('noise_double',['enzyme','forced_damping'],48,.10),
            ('near_zero',['reaction','duffing'],64,.10),
            ('fine_grid',['allen_cahn'],64,.03),
            ('missing_cubic',['reaction'],64,.10)]
    tasks=[]
    for gi,(scenario,cases,n,sigma) in enumerate(groups):
        for ci,c in enumerate(cases):
            tasks +=[(c,91000000+gi*1000000+ci*100000+i,sigma,120,5,scenario,['D_OPT','STRUCT_RISK']) for i in range(n)]
    rows=[];tic=time.perf_counter()
    with ProcessPoolExecutor(2) as pool:
        for i,res in enumerate(pool.map(run_one,tasks,chunksize=1)):
            rows.extend(res)
            if (i+1)%32==0:print('STRESS',i+1,len(tasks),round(time.perf_counter()-tic,1),flush=True)
    summarize(rows,ROOT/'stress'/'robustness')
    (ROOT/'stress'/'robustness'/'runtime.json').write_text(json.dumps(dict(independent_tasks=len(tasks),seconds=time.perf_counter()-tic),indent=2))
