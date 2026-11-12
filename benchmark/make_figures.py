from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
NAMES={'reaction':'Linear / quadratic reaction','chemostat':'Controlled co-reactant','reversible':'Reversible reaction','enzyme':'Linear / saturating kinetics','oscillator':'Amplitude dynamics','duffing':'Duffing restoring forces','forced_damping':'Forced nonlinear damping','predator':'Coupled predator-prey','diffusion_pde':'Second / fourth-order diffusion','advection_diffusion':'Advection / diffusion','allen_cahn':'Nonlinear reaction-diffusion'}
ORDER=list(NAMES)
r=pd.read_csv(ROOT/'case_results.csv')
fig,ax=plt.subplots(figsize=(10,7))
y=np.arange(len(ORDER))
for i,(p,label) in enumerate([('D_OPT','A: D-optimal'),('STRUCT_RISK','B: Bayesian structure risk')]):
    q=r[r.policy==p].set_index('case').loc[ORDER]
    bars=ax.barh(y+(i-.5)*.34,q.accuracy*100,.34,label=label)
    ax.bar_label(bars,fmt='%.1f',padding=2,fontsize=8)
ax.set_yticks(y,[NAMES[c] for c in ORDER]);ax.invert_yaxis();ax.set_xlim(0,110)
ax.set_xlabel('Exact equation-support recovery (%)')
ax.set_title('11 dynamical families; 384 independent tasks per family')
ax.legend(loc='lower center',bbox_to_anchor=(.5,-.20),ncol=2);fig.tight_layout();fig.savefig(OUT/'case_accuracy.png',dpi=180);plt.close(fig)

tr=pd.read_csv(ROOT/'replication'/'trials.csv')
fig,ax=plt.subplots(figsize=(8,4.8))
for p,label in [('RANDOM','Random'),('FIXED','Prior-optimized fixed'),('D_OPT','A: D-optimal'),('STRUCT_RISK','B: Structure risk')]:
    a=tr[tr.policy==p].groupby('round').correct.mean()
    ax.plot(a.index,100*a.values,marker='o',label=label)
ax.set_xticks([0,1,2,3]);ax.set_ylim(0,103);ax.set_xlabel('Added experiments; fixed duration and sample count')
ax.set_ylabel('Exact structure recovery (%)');ax.set_title('Paired validation on 4,224 simulated tasks');ax.legend();fig.tight_layout();fig.savefig(OUT/'learning_curve.png',dpi=180);plt.close(fig)

for case in ['enzyme','duffing','forced_damping','allen_cahn']:
    p=ROOT/'examples'/f'{case}.npz'
    if not p.exists():continue
    a=np.load(p);t=a['dense_times'];nt=len(t);nc=a['curves'].shape[1]//nt
    curves=a['curves'].reshape(-1,nt,nc);obs=a['observed'].reshape(len(a['times']),nc)
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    ax.scatter(a['times'],obs[:,0],s=22,label='New noisy observations')
    for j,line in enumerate(curves):
        ax.plot(t,line[:,0],linestyle='-' if j==len(curves)-1 else '--',label=('Updated equation' if j==len(curves)-1 else f'Old equation M{j} (frozen)'))
    ax.set_xlim(0,1.5);ax.set_xlabel('Time (same horizon for all runs)');ax.set_ylabel('Observed state, channel 1')
    ax.set_title(f'{NAMES[case]}: change the initial condition / input, not time')
    ax.legend();fig.tight_layout();fig.savefig(OUT/f'{case}_example.png',dpi=180);plt.close(fig)
print('Saved 6 figures in',OUT)
