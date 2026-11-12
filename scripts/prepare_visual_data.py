"""Build provenance-backed visual records. No new benchmark result is claimed.

Campaign animations replay archived outcomes. Three separate, noiseless STLSQ
controls isolate non-identifiability, using exact derivatives deliberately.
The latter are NOT a PySINDy/E-SINDy/UQ-SINDy benchmark.
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmark'))
from models import TIMES, T, MENUS, INITIAL, response
from experiment import obs


def stlsq(A, b, threshold=0.05, max_iter=20):
    """Transparent, unregularized sequential thresholded least squares.

    No library normalization or threshold tuning. Supplied exact derivatives
    make this a best-case structural-ambiguity diagnostic, not a noise test.
    """
    A=np.asarray(A, dtype=float); b=np.asarray(b, dtype=float)
    coef=np.linalg.lstsq(A,b,rcond=None)[0]
    for _ in range(max_iter):
        active=np.abs(coef)>=threshold
        new=np.zeros_like(coef)
        if active.any(): new[active]=np.linalg.lstsq(A[:,active],b,rcond=None)[0]
        if np.allclose(new,coef,rtol=1e-13,atol=1e-13): break
        coef=new
    return new


def rmse(a,b):return float(np.sqrt(np.mean((np.asarray(a)-b)**2)))


def controls():
    dense=np.linspace(0,3,301); t=TIMES
    out={}
    # Two decay pathways are collinear when the buffered co-reactant is fixed.
    x=np.exp(-.9*t); A=np.column_stack([-x,-x]); dy=-.9*x
    fitted=stlsq(A,dy)
    c=.1; x2=np.exp(-.9*c*t); A2=np.column_stack([-x2,-c*x2])
    updated=stlsq(np.vstack([A,A2]),np.r_[dy,-.9*c*x2])
    predict=lambda th,c: np.exp(-(th[0]+th[1]*c)*dense)
    out['co_reactant']=dict(
        title='A fixed co-reactant hides the reaction pathway',
        initial_label='Buffered co-reactant c = 1',new_label='Change c to 0.1; same observation time',
        y_label='Reactant concentration',coef_names=['a','b'],true_coef=[0,.9],
        old_coef=fitted.tolist(),new_coef=updated.tolist(),rank_before=int(np.linalg.matrix_rank(A)),rank_after=int(np.linalg.matrix_rank(np.vstack([A,A2]))),
        true_initial=predict([0,.9],1).tolist(),old_initial=predict(fitted,1).tolist(),
        true_new=predict([0,.9],c).tolist(),old_new=predict(fitted,c).tolist(),updated_new=predict(updated,c).tolist(),
        observation_initial=x.tolist(),observation_new=x2.tolist(),
        initial_equation="x' = -a x - b c x",explanation='At c = 1, the data identify only a + b.',
        intervention='Changing c separates the two library columns.',
        before_residual=float(np.linalg.norm(A@fitted-dy)),
        new_condition_rmse=rmse(predict(fitted,c),predict([0,.9],c)))
    # One spatial mode cannot distinguish second- and fourth-order damping.
    amp=np.exp(-.15*t); A=np.column_stack([-amp,-amp]); dy=-.15*amp
    fitted=stlsq(A,dy); k=2; amp2=np.exp(-.15*k*k*t)
    A2=np.column_stack([-k*k*amp2,-k**4*amp2]); updated=stlsq(np.vstack([A,A2]),np.r_[dy,-.15*k*k*amp2])
    predict=lambda th,k:np.exp(-(th[0]*k*k+th[1]*k**4)*dense)
    out['pde_mode']=dict(
        title='One spatial mode hides the diffusion order',
        initial_label='Initial field: cos(x)',new_label='Change initial field to cos(2x); same time',
        y_label='Observed Fourier-mode amplitude',coef_names=['a','b'],true_coef=[.15,0],
        old_coef=fitted.tolist(),new_coef=updated.tolist(),rank_before=int(np.linalg.matrix_rank(A)),rank_after=int(np.linalg.matrix_rank(np.vstack([A,A2]))),
        true_initial=predict([.15,0],1).tolist(),old_initial=predict(fitted,1).tolist(),
        true_new=predict([.15,0],k).tolist(),old_new=predict(fitted,k).tolist(),updated_new=predict(updated,k).tolist(),
        observation_initial=amp.tolist(),observation_new=amp2.tolist(),initial_equation='u_t = a u_xx - b u_xxxx',
        explanation='On cos(x), u_xx and -u_xxxx are identical library columns.',
        intervention='A second spatial mode scales these terms differently.',before_residual=float(np.linalg.norm(A@fitted-dy)),
        new_condition_rmse=rmse(predict(fitted,k),predict([.15,0],k)))
    # Observing a limit cycle alone misses radial attraction completely.
    r=np.ones_like(t); A=np.column_stack([r,-r**3]); dy=np.zeros_like(r)
    fitted=stlsq(A,dy); r0=.3
    radius=lambda th,tt,r0: r0*np.exp(th[0]*tt)/np.sqrt(1+(th[1]/th[0] if abs(th[0])>1e-12 else 0)*r0*r0*np.expm1(2*th[0]*tt)) if abs(th[0])>1e-12 else r0/np.sqrt(1+2*th[1]*r0*r0*tt)
    truth=np.array([.45,.45]); r2=radius(truth,t,r0); A2=np.column_stack([r2,-r2**3]); dy2=.45*r2-.45*r2**3
    updated=stlsq(np.vstack([A,A2]),np.r_[dy,dy2])
    out['limit_cycle']=dict(
        title='A limit cycle hides the restoring mechanism',
        initial_label='Start on the cycle: r(0) = 1',new_label='Start off the cycle: r(0) = 0.3; same time',
        y_label='Oscillation amplitude r',coef_names=['a','b'],true_coef=truth.tolist(),old_coef=fitted.tolist(),new_coef=updated.tolist(),
        rank_before=int(np.linalg.matrix_rank(A)),rank_after=int(np.linalg.matrix_rank(np.vstack([A,A2]))),
        true_initial=radius(truth,dense,1).tolist(),old_initial=radius(fitted,dense,1).tolist(),
        true_new=radius(truth,dense,r0).tolist(),old_new=radius(fitted,dense,r0).tolist(),updated_new=radius(updated,dense,r0).tolist(),
        observation_initial=r.tolist(),observation_new=r2.tolist(),initial_equation="r' = a r - b r^3  (known rotation omitted)",
        explanation='On r = 1, radial growth and saturation cancel exactly.',
        intervention='An off-cycle initial state reveals radial recovery.',before_residual=float(np.linalg.norm(A@fitted-dy)),
        new_condition_rmse=rmse(radius(fitted,dense,r0),radius(truth,dense,r0)))
    for v in out.values():
        v.update(times=t.tolist(),dense_times=dense.tolist(),sample_end=T,
                 provenance='New deterministic STLSQ mechanism control; exact noiseless derivatives; NOT part of the 4,224-task benchmark',
                 threshold=.05)
        assert v['rank_before']==1 and v['rank_after']==2
        assert np.allclose(v['new_coef'],v['true_coef'],atol=1e-10)
        assert np.allclose(v['old_initial'],v['true_initial'],atol=1e-10)
    return out


def make_replay(df,case,seed,policy):
    rows=df[(df.case==case)&(df.seed==seed)&(df.policy==policy)].sort_values('round')
    hit=rows[rows.confident==1]
    stop=int(hit['round'].iloc[0]) if len(hit) else 3
    rows=rows[rows['round']<=stop]
    fits=[json.loads(r.estimate) for _,r in rows.iterrows()]
    truth=np.asarray(json.loads(rows.iloc[0].truth));sigma=float(rows.iloc[0].sigma)
    dense=np.linspace(0,3,241)
    experiments=[]
    for j in range(len(rows)):
        if j==0: action=-1;stage=-1;setting=INITIAL[case]
        else:
            prev=rows.iloc[j-1];action=int(prev.action);stage=j-1;setting=MENUS[case][action]
        y=obs(case,truth,action,sigma,seed,stage)
        curves=[response(case,th,setting,times=dense,substeps=4)[0].reshape(len(dense),-1).tolist() for th in fits]
        experiments.append(dict(action=action,setting=list(setting),observed=y.reshape(len(TIMES),-1).tolist(),curves=curves))
    archive=ROOT/'benchmark'/'examples'/f'{case}.npz'
    if archive.exists() and stop==1:
        saved=np.load(archive)
        if np.array_equal(experiments[1]['setting'],saved['setting']):
            assert np.allclose(np.array(experiments[1]['observed']).ravel(),saved['observed'])
    trace=[]
    for _,r in rows.iterrows():
        trace.append(dict(round=int(r['round']),coefficients=json.loads(r.estimate),probabilities=json.loads(r.probabilities),
                          action=None if r['round']==stop else int(r.action),
                          scores=None if r['round']==stop else json.loads(r.scores)))
    return dict(case=case,seed=seed,policy=policy,sigma=sigma,stop_round=stop,times=TIMES.tolist(),dense_times=dense.tolist(),
                sample_end=T,plot_end=3,menu=[list(m) for m in MENUS[case]],trace=trace,experiments=experiments,
                provenance='Replay of archived synthetic measurements and fitted coefficients. No new campaign run; no refitting. Curves beyond t=1.5 are predictions only.',
                selection_rule='Previously archived: lowest seed initially wrong and uncertain, eventually correct. Illustrative, not an unbiased performance sample.')


def main():
    out=ROOT/'artifacts';out.mkdir(exist_ok=True)
    (out/'failure_controls.json').write_text(json.dumps(controls(),indent=2)+'\n')
    src=ROOT/'benchmark'/'replication'/'trials.csv'
    if not src.exists(): src=src.with_suffix('.csv.gz')
    df=pd.read_csv(src)
    replays=[]
    for c,s in [('enzyme',64000000),('duffing',66000007),('allen_cahn',71000000)]:
        for policy in ['D_OPT','STRUCT_RISK']:
            r=make_replay(df,c,s,policy)
            (out/f'{c}_{policy.lower()}_replay.json').write_text(json.dumps(r)+'\n')
            replays.append({k:r[k] for k in ['case','seed','policy','stop_round','provenance','selection_rule']})
    (out/'visual_provenance.json').write_text(json.dumps(dict(replays=replays,controls='failure_controls.json',
        axes='Every response curve extends to t=3. Only t<=1.5 is observed. Previous coefficients stay frozen; under a new condition the same old equation is re-simulated.',
        benchmark_source='benchmark/replication/trials.csv.gz'),indent=2)+'\n')
    print('Prepared six archived replays and three deterministic STLSQ controls.')

if __name__=='__main__':main()
