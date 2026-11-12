"""Validate the distribution without rerunning the full benchmark."""
from pathlib import Path
import gzip,hashlib,json,re,sys
import numpy as np
import pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]


def main():
    checks=[];failures=[]
    def check(name,ok,detail=''):
        checks.append(dict(name=name,passed=bool(ok),detail=detail))
        if not ok:failures.append(name+': '+str(detail))
    # Publication artifact integrity; regenerated outputs are described in docs/PROVENANCE.md.
    manifest=json.loads((ROOT/'benchmark/ARTIFACT_SHA256.json').read_text())
    mismatch=[]
    for name,expected in manifest.items():
        p=ROOT/'benchmark'/name
        if p.exists():data=p.read_bytes()
        elif p.with_suffix(p.suffix+'.gz').exists():data=gzip.decompress(p.with_suffix(p.suffix+'.gz').read_bytes())
        else:mismatch.append(name+' missing');continue
        if hashlib.sha256(data).hexdigest()!=expected:mismatch.append(name+' changed')
    check('Publication benchmark artifacts match release hashes',not mismatch,{'files':len(manifest),'mismatch':mismatch})
    p=ROOT/'benchmark/replication/trials.csv.gz'
    if not p.exists():p=p.with_suffix('')
    d=pd.read_csv(p);fin=d[d['round']==3]
    check('4,224 independent main tasks',len(d[['case','seed','sigma']].drop_duplicates())==4224)
    check('Four paired policies per task',len(fin)==4224*4)
    check('All task paths have four rounds',d.groupby(['case','seed','sigma','policy']).size().eq(4).all())
    expected=pd.read_csv(ROOT/'benchmark/overall.csv');expected=expected[expected['subset']=='all'].set_index('policy').accuracy
    calc=fin.groupby('policy').correct.mean()
    check('Published accuracy equals raw records',np.allclose(calc.loc[expected.index],expected,atol=1e-14))
    st=d[d.stopped_here==1];check('One first-stop record per paired path',len(st)==4224*4)
    check('Stops use the declared threshold or budget',((st.confident==1)|(st['round']==3)).all())
    # Validate all local links in new user-facing Markdown (external references are not fetched here).
    missing=[];paths=[ROOT/'README.md',*sorted((ROOT/'docs').glob('*.md'))]
    for md in paths:
        for url in re.findall(r'\]\(([^)]+)\)',md.read_text()):
            if re.match(r'^(https?://|mailto:|#)',url):continue
            url=url.split('#')[0]
            if url and not (md.parent/url).exists():missing.append(f'{md.relative_to(ROOT)} -> {url}')
    check('All local README/document links resolve',not missing,missing)
# Check all readable source and documentation files for Han characters.
    nonenglish=[]
    suffixes={'.md','.py','.txt','.json','.csv','.yml','.yaml','.cff','.toml','.ini'}
    for source in ROOT.rglob('*'):
        if source.is_file() and source.suffix.lower() in suffixes:
            try:
                content=source.read_text(encoding='utf-8')
            except UnicodeError:
                continue
            if any(0x3400 <= ord(ch) <= 0x9fff for ch in content):
                nonenglish.append(str(source.relative_to(ROOT)))
    check('All repository text is English-only',not nonenglish,nonenglish)
    images={}
    for p in sorted((ROOT/'assets').glob('*.gif')):
        with Image.open(p) as im:
            duration=0
            for i in range(im.n_frames):im.seek(i);duration+=im.info.get('duration',0)
            images[p.name]={'frames':im.n_frames,'seconds':duration/1000,'size':list(im.size)}
            check('Animated '+p.name,im.n_frames>1 and duration>0)
    check('Five README GIFs present',len(images)==5)
    ctrl=json.loads((ROOT/'artifacts/failure_controls.json').read_text())
    check('Three honest noise-free STLSQ controls',len(ctrl)==3 and all(c['rank_before']==1 and c['rank_after']==2 and np.allclose(c['new_coef'],c['true_coef'],atol=1e-10) for c in ctrl.values()))
    replay_files=list((ROOT/'artifacts').glob('*_replay.json'))
    check('Six auditable replay records',len(replay_files)==6)
    for p in replay_files:
        r=json.loads(p.read_text());q=d[(d.case==r['case'])&(d.seed==r['seed'])&(d.policy==r['policy'])].sort_values('round')
        same=True
        for row in r['trace']:
            ar=q[q['round']==row['round']].iloc[0]
            same &= np.allclose(row['coefficients'],json.loads(ar.estimate))
            same &= np.allclose(row['probabilities'],json.loads(ar.probabilities))
            if row['scores'] is not None:same &= np.allclose(row['scores'],json.loads(ar.scores))
        check('Archived fits and design scores: '+p.stem,same)
    report=dict(passed=not failures,checks=checks,gifs=images,
                scope='Publication reconstruction checks; the original seeds are reused, not an independent new population.')
    (ROOT/'artifacts/package_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':not failures,'checks':len(checks),'failures':failures,'main_tasks':4224,'gifs':len(images)},indent=2))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
