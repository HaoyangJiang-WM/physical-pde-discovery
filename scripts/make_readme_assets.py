"""Render factual README figures/GIFs from prepared, auditable records.

Each Matplotlib chart is rendered separately; Pillow assembles presentation
panels. Colors use Matplotlib defaults. No generative images or fake data.
"""
import io,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/active-equation-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets';OUT.mkdir(exist_ok=True)
from matplotlib.font_manager import FontProperties, findfont
FONT=findfont(FontProperties(family='DejaVu Sans'))
BOLD=findfont(FontProperties(family='DejaVu Sans',weight='bold'))
W,H=1200,795


def font(n,bold=False):return ImageFont.truetype(BOLD if bold else FONT,n)
def text(draw,xy,s,n=21,bold=False):draw.text(xy,str(s),font=font(n,bold),fill='black')
def wrap(draw,s,width,n=19,bold=False):
    lines=[];line=''
    for word in s.split():
        test=f'{line} {word}'.strip()
        if draw.textlength(test,font=font(n,bold))>width and line:lines.append(line);line=word
        else:line=test
    return lines+[line]
def block(draw,xy,s,width,n=19,bold=False,spacing=5):
    x,y=xy
    for line in wrap(draw,s,width,n,bold):text(draw,(x,y),line,n,bold);y+=n+spacing
    return y

def fig_image(fig):
    fig.canvas.draw();a=np.asarray(fig.canvas.buffer_rgba()).copy();plt.close(fig)
    return Image.fromarray(a[:,:,:3])

def equation(case,th):
    a,b=th
    terms=[]
    if case=='enzyme':
        if abs(a)>1e-8:terms.append(f'-{a:.4f} x')
        if abs(b)>1e-8:terms.append(f'-{b:.4f} x/(0.4+x)')
        return "x' = "+(' '.join(terms) if terms else '0')
    if case=='duffing':
        return "v' = -0.3 v"+(f' - {a:.4f} x' if abs(a)>1e-8 else '')+(f' - {b:.4f} x^3' if abs(b)>1e-8 else '')
    return 'u_t = 0.04 u_xx'+(f' + {a:.4f} u' if abs(a)>1e-8 else '')+(f' - {b:.4f} u^3' if abs(b)>1e-8 else '')

def condition(case,s):
    if case=='enzyme':return f'Initial concentration x(0) = {s[0]:g}'
    if case=='duffing':return f'Initial amplitude x(0) = {s[0]:g}'
    return f'Initial field = {s[0]:g} cos({s[1]:g}x)'

def trajectory(r,exp,show_fits,visible=16):
    fig,ax=plt.subplots(figsize=(5.55,3.85),dpi=100)
    e=r['experiments'][exp];t=np.array(r['dense_times']);tm=np.array(r['times'])
    curves=[np.array(a)[:,0] for a in e['curves']]
    for j in range(show_fits):
        ax.plot(t,curves[j],linestyle='--' if j==0 else '-',linewidth=2.2,
                label='Old equation (frozen)' if j==0 else f'Updated equation M{j}')
    if visible:
        y=np.array(e['observed'])[:,0]
        ax.scatter(tm[:visible],y[:visible],s=30,label='Acquired measurements',zorder=5)
    ax.axvline(r['sample_end'],linestyle=':',linewidth=1)
    ax.text(.72,.96,'Prediction only',transform=ax.transAxes,ha='center',va='top',fontsize=9)
    ax.set_xlim(0,3)
    all_y=np.r_[np.array(e['observed'])[:,0],*[c for c in curves]]
    span=max(.1,np.ptp(all_y));ax.set_ylim(all_y.min()-.1*span,all_y.max()+.18*span)
    ax.set_xlabel('Time (all acquisitions end at 1.5)')
    ax.set_ylabel('Concentration' if r['case']=='enzyme' else 'Displacement' if r['case']=='duffing' else 'Field value at observed site x = 0')
    ax.set_title(condition(r['case'],e['setting']),fontsize=11)
    if show_fits or visible:ax.legend(fontsize=8,loc='best')
    fig.tight_layout(pad=1.25)
    return fig_image(fig)

def score_plot(r):
    fig,ax=plt.subplots(figsize=(5.55,3.85),dpi=100)
    row=r['trace'][0];scores=np.array(row['scores']); labels=[condition(r['case'],s).replace('Initial concentration ','').replace('Initial amplitude ','').replace('Initial field = ','') for s in r['menu']]
    bars=ax.barh(labels,scores)
    ax.bar_label(bars,fmt='%.3g',padding=4,fontsize=9)
    ax.set_xlim(0,max(scores)*1.25+1e-10)
    ax.invert_yaxis();ax.set_xlabel('Expected log-det gain' if r['policy']=='D_OPT' else 'Expected reduction in structure error')
    ax.set_title('Score candidates BEFORE collecting new data',fontsize=11)
    for j,lab in enumerate(ax.get_yticklabels()):
        if j==row['action']:lab.set_fontweight('bold')
    fig.tight_layout(pad=1.25);return fig_image(fig)

def save_gif(name,frames,durations):
    frames[0].save(OUT/name,save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=True,disposal=2)

def campaign(r,filename):
    assert r['stop_round']==1,'These curated animations show a one-intervention example.'
    method='A | D-optimal' if r['policy']=='D_OPT' else 'B | Bayesian structure risk'
    title={'enzyme':'Change the concentration. Discover the missing mechanism.','duffing':'A new amplitude reveals nonlinear stiffness.','allen_cahn':'A new initial field reveals a nonlinear PDE term.'}[r['case']]
    captions=[
      ('1. OBSERVE','Only the existing 16-point experiment is available.',0,0,16,0),
      ('2. FIT','A sparse equation fits the old condition. That does not establish a unique mechanism.',1,0,16,0),
      ('3. DESIGN','Use the fitted equation and all candidate mechanisms to score new conditions.',1,-1,16,0),
      ('4. PREDICT','Choose a different condition. The dashed curve uses the OLD coefficients, unchanged.',1,1,16,0),
      ('5. MEASURE','Run the selected synthetic system. New observations are NOT model-generated labels.',1,1,16,8),
      ('5. MEASURE','The new response challenges the old equation. Acquisition time is still 1.5.',1,1,16,16),
      ('6. REFIT','Combine both experiments using one shared set of physical coefficients.',2,2,16,16),
      ('7. STOP','The structure posterior crosses 95%. Do not spend the remaining budget just to continue.',2,2,16,16)]
    frames=[]
    for k,(stage,desc,leftfit,rightfit,lv,rv) in enumerate(captions):
        im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im)
        text(d,(34,22),'ACTIVE EQUATION DISCOVERY',18,True)
        text(d,(34,55),title,27,True)
        text(d,(34,100),f'{method}   /   Fixed duration, fixed samples, changed condition',19)
        text(d,(34,141),stage,23,True);block(d,(34,177),desc,W-68,20)
        im.paste(trajectory(r,0,leftfit,lv),(30,236))
        if rightfit==-1:im.paste(score_plot(r),(615,236))
        elif rightfit>0:im.paste(trajectory(r,1,rightfit,rv),(615,236))
        else:
            d.rectangle((638,270,1150,555),outline='black',width=1)
            block(d,(665,320),'No new experiment has been observed yet.',450,25,True)
            block(d,(665,411),'The next condition must be chosen without looking at its outcomes.',450,21)
        d=ImageDraw.Draw(im)
        text(d,(36,637),'M0: '+equation(r['case'],r['trace'][0]['coefficients']),20)
        if k>=6:
            text(d,(36,674),'M1: '+equation(r['case'],r['trace'][1]['coefficients']),20,True)
            p=max(r['trace'][1]['probabilities']);text(d,(36,710),f'Posterior of selected structure: {100*p:.3f}%  |  1 added experiment',19)
        else:
            p=max(r['trace'][0]['probabilities']);text(d,(36,680),f'Initial maximum structure posterior: {100*p:.1f}%  |  still ambiguous',19)
        text(d,(36,758),'Archived synthetic replay; selected illustrative success, not a benchmark average.',15)
        frames.append(im)
    save_gif(filename,frames,[1800,2500,3400,2500,1000,2000,3200,3800])
    frames[-1].save(OUT/(filename.replace('.gif','_poster.png')))
    return frames

def control_chart(c,new=False,updated=False):
    fig,ax=plt.subplots(figsize=(5.55,3.85),dpi=100)
    t=np.array(c['dense_times']);tm=np.array(c['times'])
    key='new' if new else 'initial'
    ax.plot(t,c['true_'+key],linewidth=2,label='Generating equation')
    ax.plot(t,c['old_'+key],linestyle='--',linewidth=2,label='STLSQ fit from old data')
    if updated:ax.plot(t,c['updated_new'],linestyle=':',linewidth=2.5,label='Refit after intervention')
    ax.scatter(tm,c['observation_'+key],s=22,label='Observed states (noise-free)',zorder=5)
    ax.axvline(1.5,linestyle=':',linewidth=1);ax.text(.75,.95,'Prediction only',ha='center',va='top',transform=ax.transAxes,fontsize=9)
    ax.set_xlim(0,3);ax.set_xlabel('Time');ax.set_ylabel(c['y_label'])
    ax.set_title(c['new_label'] if new else c['initial_label'],fontsize=10)
    ax.legend(fontsize=8,loc='best');fig.tight_layout(pad=1.2);return fig_image(fig)

def failure_assets():
    cs=json.loads((ROOT/'artifacts/failure_controls.json').read_text());frames=[];durations=[]
    for i,(name,c) in enumerate(cs.items()):
        for stage in range(3):
            im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im)
            text(d,(34,25),f'WHY FITTING ALONE CAN BE INSUFFICIENT  /  CASE {i+1}',18,True)
            text(d,(34,66),c['title'],28,True)
            text(d,(34,112),c['initial_equation'],23)
            block(d,(34,156),c['explanation'],W-68,21)
            im.paste(control_chart(c),(30,229))
            if stage>0:im.paste(control_chart(c,True,stage==2),(615,229))
            else:
                d=ImageDraw.Draw(im);block(d,(658,320),'An exact fit is not an identifiability certificate.',460,26,True)
                block(d,(658,420),'Even noise-free data and exact derivatives leave two mechanisms confounded.',460,21)
            d=ImageDraw.Draw(im)
            text(d,(34,640),['FIT: different coefficients reproduce the same observed trajectory.',
                            'INTERVENE: change the condition, not the acquisition time.',
                            'REFIT: require one coefficient vector to explain both experiments.'][stage],20,True)
            block(d,(34,681),c['intervention']+f"   Library rank: {c['rank_before']} -> {c['rank_after']}.",W-68,19)
            text(d,(34,757),'Deterministic STLSQ diagnostic with exact derivatives; not part of the archived benchmark.',15)
            frames.append(im);durations.append(2600 if stage<2 else 3200)
            if stage==2:im.save(OUT/f'failure_{name}.png')
    save_gif('why_intervene.gif',frames,durations)


def result_assets():
    d=pd.read_csv(ROOT/'benchmark/case_results.csv')
    cases=['reaction','chemostat','reversible','enzyme','oscillator','duffing','forced_damping','predator','diffusion_pde','advection_diffusion','allen_cahn']
    labels=['Reaction order','Buffered co-reactant','Reversible reaction','Saturating kinetics','Amplitude dynamics','Duffing stiffness','Forced nonlinear damping','Two-species interaction','Diffusion order','Advection-diffusion','Nonlinear reaction-diffusion']
    fig,ax=plt.subplots(figsize=(9,6.4),dpi=150)
    y=np.arange(len(cases))
    for j,(p,l) in enumerate([('D_OPT','A: D-optimal'),('STRUCT_RISK','B: Bayesian structure risk')]):
        v=d[d.policy==p].set_index('case').loc[cases,'accuracy'].to_numpy()*100
        b=ax.barh(y+(j-.5)*.32,v,.32,label=l);ax.bar_label(b,fmt='%.1f',fontsize=8,padding=3)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,108)
    ax.set_xlabel('Exact support recovery after the full three-experiment budget (%)')
    ax.set_title('11 synthetic families | 384 independent tasks per family')
    ax.legend(loc='lower left',fontsize=9);fig.tight_layout();fig.savefig(OUT/'benchmark_results.png');plt.close(fig)
    # One learning-curve chart, no invented intermediate results.
    tr=pd.read_csv(ROOT/'benchmark/learning_curves.csv')
    # Derive equal-task means from the archived summary, preserving sample sizes.
    raw=ROOT/'benchmark/replication/trials.csv'
    if not raw.exists():raw=raw.with_suffix('.csv.gz')
    df=pd.read_csv(raw,usecols=['policy','round','correct'])
    fig,ax=plt.subplots(figsize=(8,4.4),dpi=150)
    for p,l in [('RANDOM','Random'),('FIXED','Prior-optimized fixed'),('D_OPT','A: D-optimal'),('STRUCT_RISK','B: Structure risk')]:
        s=df[df.policy==p].groupby('round')['correct'].mean()*100
        ax.plot(s.index,s.values,marker='o',label=l)
    ax.set_xticks(range(4));ax.set_ylim(0,103);ax.set_xlabel('Additional experiments (same duration and samples)')
    ax.set_ylabel('Exact support recovery (%)');ax.legend();fig.tight_layout();fig.savefig(OUT/'learning_curve.png');plt.close(fig)


def main():
    frames={}
    for c,p,name in [('enzyme','STRUCT_RISK','bayesian_loop.gif'),('enzyme','D_OPT','doptimal_loop.gif'),('duffing','STRUCT_RISK','duffing_loop.gif'),('allen_cahn','STRUCT_RISK','pde_loop.gif')]:
        r=json.loads((ROOT/f'artifacts/{c}_{p.lower()}_replay.json').read_text())
        frames[name]=campaign(r,name)
    failure_assets();result_assets()
    contact=Image.new('RGB',(W*2,H*2),'white')
    for idx,f in enumerate([frames['bayesian_loop.gif'][2],frames['bayesian_loop.gif'][-1],frames['duffing_loop.gif'][-1],frames['pde_loop.gif'][-1]]):
        contact.paste(f,((idx%2)*W,(idx//2)*H))
    contact.resize((1200,795)).save(ROOT/'artifacts/visual_contact_sheet.png')
    metadata={}
    for p in sorted(OUT.glob('*.gif')):
        with Image.open(p) as im:
            durations=[]
            for i in range(im.n_frames):im.seek(i);durations.append(im.info.get('duration',0))
            metadata[p.name]=dict(frames=im.n_frames,size=list(im.size),duration_seconds=sum(durations)/1000,bytes=p.stat().st_size)
    (ROOT/'artifacts/gif_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
