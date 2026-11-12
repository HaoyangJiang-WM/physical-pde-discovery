"""Parallel, recoverable publication of the frozen research artifact."""
from pathlib import Path
import os, shutil, sys

phase=sys.argv[1]
source=Path('.publish/restore.py').read_text()
# Preserve only the immutable, originally transferred core. Each job runs the
# same source lock and main-branch guard as the initial publication attempt.
prefix=source[:source.index('# Rebuild binary records')]
common=source[:source.index("run('git','fetch','origin','main')")]
final=source[source.index("run(sys.executable,'analyze.py',cwd=B)"):]
for name in ('audit_solvers.py','audit_inference.py','audit_reaction3.py'):
    final=final.replace("run(sys.executable,'%s',cwd=B)\n"%name,'')
# The checker links to its own generated report. Make its status explicit until
# the checker overwrites it with the actual 22-check result.
final=final.replace("run(sys.executable,'run.py','check')", "write('artifacts/package_validation.json',json.dumps({'status':'validation pending; this file is replaced by the actual check results'})+'\\n')\nrun(sys.executable,'run.py','check')")
final_path=Path(os.environ.get('RUNNER_TEMP','/tmp'))/'active-finalize.py'
final_path.write_text(common+"\nB=ROOT/'benchmark'\n"+final)
ns={'__name__':'publication_core'}
exec(compile(prefix,'frozen_publication_core.py','exec'),ns)
ROOT=ns['ROOT'];B=ns['B'];run=ns['run'];protocol=ns['protocol']
if phase=='prepare':
    print('Core prepared; finalizer:',final_path,flush=True)
    raise SystemExit(0)
if phase=='main':
    protocol('replication')
    import numpy as np,pandas as pd
    raw=pd.read_csv(B/'replication/trials.csv')
    expected=pd.read_csv(ROOT/'artifacts/original_overall.csv').query("subset == 'all'").set_index('policy')['accuracy']
    actual=raw[raw['round']==3].groupby('policy')['correct'].mean()
    assert np.allclose(actual.loc[expected.index],expected,atol=1e-14),'Main accuracy differs from original reference'
    assert len(raw[['case','seed','sigma']].drop_duplicates())==4224
    protocol('development')
    run(sys.executable,'audit_solvers.py',cwd=B)
    run(sys.executable,'audit_inference.py',cwd=B)
    directories=['replication','development','audits']
elif phase=='extras':
    protocol('stress/reaction3')
    run(sys.executable,'run_stress.py',cwd=B)
    run(sys.executable,'noise_aware.py',cwd=B)
    run(sys.executable,'audit_reaction3.py',cwd=B)
    directories=['stress','audits']
else:
    raise ValueError('Unknown publication phase')
payload=ROOT/'publication-payload'/'benchmark'
payload.mkdir(parents=True,exist_ok=True)
for name in directories:
    shutil.copytree(B/name,payload/name,dirs_exist_ok=True)
print('Reconstruction artifact ready:',phase,flush=True)
