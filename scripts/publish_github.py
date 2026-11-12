#!/usr/bin/env python3
"""Publish this artifact as a NEW repository using the user's GitHub CLI.

Nothing is published when this file is merely imported. Visibility is mandatory.
Refuses an existing target. No remote, branch or file in another repo is modified.
"""
from pathlib import Path
import argparse
import re
import shutil
import subprocess
import sys


def run(*args,check=True):
    return subprocess.run(args,check=check,text=True,capture_output=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('repository',help='OWNER/NEW-REPOSITORY')
    visibility=ap.add_mutually_exclusive_group(required=True)
    visibility.add_argument('--private',action='store_true')
    visibility.add_argument('--public',action='store_true')
    a=ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',a.repository):ap.error('Use OWNER/REPOSITORY.')
    root=Path(__file__).resolve().parents[1]
    if not shutil.which('gh') or not shutil.which('git'):ap.error('Install Git and GitHub CLI first.')
    auth=run('gh','auth','status',check=False)
    if auth.returncode:ap.error('Authenticate with gh auth login first; no token is stored here.')
    exists=run('gh','repo','view',a.repository,'--json','name',check=False)
    if exists.returncode==0:ap.error('Target already exists. Refusing to modify it.')
    subprocess.run([sys.executable,'run.py','check'],cwd=root,check=True)
    # A local Git repo may already be included as an optional bundle, but never
    # take over an unrelated parent repo. Require this exact root.
    if not (root/'.git').exists():
        subprocess.run(['git','init','-b','main'],cwd=root,check=True)
        subprocess.run(['git','add','.'],cwd=root,check=True)
        subprocess.run(['git','commit','-m','Package active equation discovery benchmark and visual README'],cwd=root,check=True)
    subprocess.run(['gh','repo','create',a.repository,'--private' if a.private else '--public',
                    '--source',str(root),'--remote','origin','--push',
                    '--description','Fixed-duration active equation discovery: sparse models, D-optimal and Bayesian structure-risk experiments.'],cwd=root,check=True)
    print('Published:',a.repository)

if __name__=='__main__':main()
