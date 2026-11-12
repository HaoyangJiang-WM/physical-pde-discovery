"""Publish a verified reconstruction without rewriting repository history."""
from pathlib import Path
import base64, gzip, hashlib, io, json, os, platform, shutil, subprocess, sys, tarfile, tempfile

ROOT=Path.cwd()
EXPECTED_PARENT='2d0719f11b37063f874aa0ca5a9dde7cdeed54ab'
EXPECTED_PAYLOAD='ef6b6e7561e4160c58aa2533105ba9acc88cf291b7c769c2e272a701d8d93403'
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
os.environ['MPLBACKEND']='Agg'
os.environ['PYTHONUNBUFFERED']='1'
os.environ['MPLCONFIGDIR']=str(Path(tempfile.gettempdir())/'active-equations-mpl')

def run(*args,cwd=None):
    print('\nRUN:', ' '.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),cwd=cwd or ROOT,check=True)

def write(path,text):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8')

run('git','fetch','origin','main')
parent=subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip()
if parent!=EXPECTED_PARENT:
    raise RuntimeError('The main branch changed while staging; refusing to replace new work.')
parts=[ROOT/'.publish'/f'part_{i:02d}.txt' for i in range(7)]
payload=base64.b64decode(''.join(p.read_text().strip() for p in parts),validate=True)
assert hashlib.sha256(payload).hexdigest()==EXPECTED_PAYLOAD,'Payload checksum failed'
stage=Path(tempfile.mkdtemp(prefix='active-equation-publication-'))
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:xz') as archive:
    members=archive.getmembers()
    for m in members:
        assert m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts
    archive.extractall(stage,filter='data')
assert len(members)==47
# No remote branch is touched until all of the following checks succeed.
for p in ROOT.iterdir():
    if p.name=='.git':continue
    if p.is_dir() and not p.is_symlink():shutil.rmtree(p)
    else:p.unlink()
shutil.copytree(stage,ROOT,dirs_exist_ok=True)
B=ROOT/'benchmark'
for name in ('artifacts','assets','benchmark/audits','benchmark/examples'):
    (ROOT/name).mkdir(parents=True,exist_ok=True)
reference_names=['case_results.csv','overall.csv','overall_stopping.csv','learning_curves.csv']
for name in reference_names:
    shutil.copyfile(B/name,ROOT/'artifacts'/('original_'+name))
run(sys.executable,'-m','pip','install','-r','requirements.txt')
# Preserve the frozen source/plan lock; these files must not change.
lock=json.loads((B/'REPLICATION_LOCK.json').read_text())
for name,sha in lock['files'].items():
    assert hashlib.sha256((B/name).read_bytes()).hexdigest()==sha,name

def protocol(name):
    cfg=json.loads((B/name/'protocol.json').read_text())
    args=[sys.executable,'experiment.py']
    for key in ('cases','n','order','power','sigmas','seed','policies','scenario'):
        args += ['--'+key,str(cfg[key])]
    args += ['--workers','4','--out',name]
    run(*args,cwd=B)

# Rebuild binary records from their original seeds: this is labeled regeneration,
# not a byte-for-byte upload of the original raw archive or new independent data.
protocol('replication')
import numpy as np
import pandas as pd
raw=pd.read_csv(B/'replication/trials.csv')
expected=pd.read_csv(ROOT/'artifacts/original_overall.csv').query("subset == 'all'").set_index('policy')['accuracy']
actual=raw[raw['round']==3].groupby('policy')['correct'].mean()
assert np.allclose(actual.loc[expected.index],expected,atol=1e-14),'Regeneration changed published main accuracy'
assert len(raw[['case','seed','sigma']].drop_duplicates())==4224
protocol('development')
protocol('stress/reaction3')
run(sys.executable,'run_stress.py',cwd=B)
run(sys.executable,'noise_aware.py',cwd=B)
run(sys.executable,'analyze.py',cwd=B)
run(sys.executable,'analyze_extra.py',cwd=B)
run(sys.executable,'audit_solvers.py',cwd=B)
run(sys.executable,'audit_inference.py',cwd=B)
run(sys.executable,'audit_reaction3.py',cwd=B)
run(sys.executable,'export_examples.py',cwd=B)
run(sys.executable,'make_figures.py',cwd=B)
run(sys.executable,'make_report.py',cwd=B)
# Generate all data-backed README assets from the reconstructed task logs.
run(sys.executable,'run.py','figures')
run(sys.executable,'-m','unittest','discover','-s','tests','-v')
run(sys.executable,'run.py','demo')
run(sys.executable,'run.py','summary')

provenance='''# Provenance and publication

This repository publishes **Active Equation Discovery**, replacing the previous heat-transfer demonstration at the owner's explicit request. The previous content remains recoverable in Git history; no history was force-rewritten. The original main commit was `2d0719f11b37063f874aa0ca5a9dde7cdeed54ab`.

## Original research and this publication

The source research artifact was `active_discovery_two_schemes_v21.zip`, SHA-256 `c73dee6144d96b2c65b8a06905b4e18b4f4262afb8a06a999573a1cddab77321`. The readable repository was subsequently packaged with English/Chinese READMEs and five numerical GIFs.

The publication transfers the frozen source, fixed plans, protocols, documentation and main reference tables. The GitHub runner **regenerates** task logs, example observations, numerical audits and graphics from the recorded seeds and protocols. This is not a byte-for-byte upload of the original numerical archive, not a new independent validation population, and not hardware data. Execution timings, environment information and byte hashes can differ. Original main summary tables are retained as `artifacts/original_*.csv`. Publication requires the reconstructed task count and main support accuracies to match the supplied reference tables.

`benchmark/REPLICATION_LOCK.json` retains the original frozen source/plan checks. `benchmark/ARTIFACT_SHA256.json` hashes the **regenerated publication files**, with transparent decompression for CSVs. It is an integrity manifest for this release, not a claim that all original output bytes are unchanged. `artifacts/publication.json` records the workflow and reconstruction scope.

## Animation rules

Each old model keeps its coefficients frozen. All model curves extend to t=3; real simulated observations stop at t=1.5. Changing the experimental condition is not extending the measurement window. Historical curves remain visible. Scores and coefficients are read from task logs. The selected successful examples illustrate mechanisms and are not estimates of average performance.

## Scope and validation

The 4,224 primary worlds, 192 three-mechanism worlds, and 672 stress worlds are separate task populations. Noise-aware runs reuse 192 stress worlds and do not count as new independent tasks. No official PySINDy/UQ-SINDy reproduction, company-internal data, real chemical experiment, universal identifiability proof, or new D-optimal/Bayesian design criterion is claimed.

The three noiseless STLSQ controls isolate lack of experimental information, rather than numerical differentiation failure. They are not part of the main benchmark counts. Unit tests, source-lock checks, reconstructed statistics, numerical audits, replay consistency and local-link/image checks are run during publication.

No font files, credentials, emails or other private documents are included. An open-source license has not been chosen on the owner's behalf.
'''
write('docs/PROVENANCE.md',provenance)
p=ROOT/'docs/REPRODUCE.md';text=p.read_text()
text=text.replace('original benchmark versions remain in `benchmark/environment.json`.','the publication benchmark environment is also recorded in `benchmark/environment.json`.')
text=text.replace('## Publish as a new repository', '## Published repository')
text=text.split('## Published repository')[0]+'''## Published repository

The project is published at https://github.com/HaoyangJiang-WM/physical-pde-discovery . Its earlier heat-transfer files are retained in Git history, not in the current tree.

This release rebuilds numerical records and media from frozen source, plans and recorded seeds. See [publication provenance](PROVENANCE.md). Main reference tables remain in `artifacts/original_*.csv`; reconstructed task counts and accuracies are checked against them. Regeneration is not another independent benchmark population. Gzip archives contain the reconstructed records, not a claim of byte identity to earlier runtime-dependent CSVs.
'''
text=text.replace('examples/                     original measurements and example histories','examples/                     reconstructed observations and example histories')
p.write_text(text)
p=ROOT/'README.zh-CN.md';text=p.read_text().replace('不是平均性能，也不是新跑的一轮实验。','不是平均性能，也不是新增的独立评估样本；发布时按原始种子重建。');p.write_text(text)
# Adapt the integrity check honestly: no fabricated original-output hashes.
p=ROOT/'scripts/check_repository.py';text=p.read_text()
text=text.replace('# Byte-for-byte original numerical archive, accepting compressed CSV storage.','# Publication artifact integrity; regenerated outputs are described in docs/PROVENANCE.md.')
text=text.replace("'Original benchmark archive unchanged'","'Publication benchmark artifacts match release hashes'")
text=text.replace('Packaging/replay validation; not a new 4,224-task benchmark run.','Publication reconstruction checks; the original seeds are reused, not an independent new population.')
p.write_text(text)
# Record actual dependencies rather than copying an earlier environment claim.
import scipy, numba, matplotlib, PIL
versions=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,numba=numba.__version__,matplotlib=matplotlib.__version__,pillow=PIL.__version__,platform=platform.platform())
write('artifacts/package_environment.json',json.dumps(versions,indent=2)+'\n')
write('benchmark/environment.json',json.dumps(versions,indent=2)+'\n')
write('artifacts/publication.json',json.dumps(dict(repository='HaoyangJiang-WM/physical-pde-discovery',original_main=EXPECTED_PARENT,source_payload_sha256=EXPECTED_PAYLOAD,run_id=os.getenv('GITHUB_RUN_ID'),raw_outputs='Regenerated from the original frozen protocols and seeds; not byte-identical archive transfer.',primary_independent_worlds=4224,three_mechanism_worlds=192,stress_worlds=672,paired_noise_reuse=192,main_accuracy_verified_against_original=True),indent=2)+'\n')
write('.github/workflows/check.yml','''name: Check research artifact
on: [push, pull_request, workflow_dispatch]
permissions:
  contents: read
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: python -m pip install -r requirements.txt
      - run: python run.py check
      - run: python -m unittest discover -s tests -v
''')
# Manifest records decompressed CSV bytes, matching the checker semantics.
manifest={}
for p in sorted(B.rglob('*')):
    if not p.is_file() or any(x in p.parts for x in ('cache','__pycache__')) or p.name=='ARTIFACT_SHA256.json':continue
    manifest[str(p.relative_to(B))]=hashlib.sha256(p.read_bytes()).hexdigest()
write('benchmark/ARTIFACT_SHA256.json',json.dumps(manifest,indent=2)+'\n')
for p in sorted(B.rglob('trials.csv')):
    compressed=p.with_suffix('.csv.gz')
    with compressed.open('wb') as fh,gzip.GzipFile(fileobj=fh,mode='wb',mtime=0,filename='') as z:
        z.write(p.read_bytes())
    p.unlink()
run(sys.executable,'run.py','check')
# Remove only generated, ignored caches; never ship credentials or font files.
for p in list(ROOT.rglob('__pycache__')):shutil.rmtree(p)
shutil.rmtree(B/'cache',ignore_errors=True)
print('\nPublication validated: README, five GIFs, source, reconstructed records and tests.',flush=True)
