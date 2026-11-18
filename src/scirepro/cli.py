"""Public command-line interface; one invocation, one immutable run directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from datetime import datetime, timezone


def file_manifest(folder):
    rows=[]
    for p in sorted(folder.rglob('*')):
        if p.is_file() and not any(x.startswith('.') for x in p.relative_to(folder).parts):
            h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):
                    h.update(block)
            rows.append({'path':str(p.relative_to(folder)),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
    return rows


def run(args):
    study=Path(args.study).expanduser().resolve()
    if not study.is_dir():
        raise ValueError(f'Study directory does not exist: {study}')
    # Set environment before importing original modules (configuration is read at import time).
    if args.model:
        os.environ['MODEL_NAME']=args.model
    if args.backend:
        os.environ['BACKEND']=args.backend
    if args.max_turns is not None:
        os.environ['MAX_AGENT_TURNS']=str(args.max_turns)
    if args.temperature is not None:
        os.environ['TEMPERATURE']=str(args.temperature)
    from . import model
    from .agent import run_agent
    from .study import prepare, discover
    info=discover(study)
    mode=args.mode
    if mode=='auto':
        mode='both' if info['code'] else 'reproduce'
    if mode in {'audit','both'} and not info['code']:
        raise ValueError('Audit requires original analysis code; use --mode reproduce or add analysis code.')
    slug=re.sub(r'[^a-zA-Z0-9_-]+','-',model.MODEL_NAME).strip('-')[:70]
    base=Path(args.output).expanduser().resolve() if args.output else (Path.cwd()/'outputs').resolve()
    if base == study or base in study.parents or study in base.parents:
        raise ValueError('Output directory must be outside the study directory.')
    base.mkdir(parents=True,exist_ok=True)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+slug
    dest=base/run_id
    dest.mkdir()
    work=dest/'study'
    shutil.copytree(study,work,ignore=shutil.ignore_patterns('.git','__pycache__','.env','outputs','logs'))
    tasks=['reproduce','audit'] if mode=='both' else [mode]
    results={}
    metadata={'model':model.MODEL_NAME,'backend':model.BACKEND,'mode':mode,'source_study':str(study),'started_utc':datetime.now(timezone.utc).isoformat(),'input_manifest':file_manifest(study),'runs':{},'settings':{'temperature':model.TEMPERATURE,'seed':model.SEED,'max_new_tokens':model.MAX_NEW_TOKENS,'max_turns':int(os.environ.get('MAX_AGENT_TURNS','20'))}}
    (dest/'manifest.json').write_text(json.dumps(metadata,indent=2))
    try:
        for task in tasks:
            # Independent fresh study copies prevent cross-task contamination.
            taskdir=dest/task
            taskdir.mkdir()
            taskstudy=taskdir/'study'
            shutil.copytree(work,taskstudy)
            details=prepare(taskstudy,task,taskstudy)
            metadata['runs'][task]={'status':'running','study_info':details}
            (dest/'manifest.json').write_text(json.dumps(metadata,indent=2))
            messages=[]
            report=''
            try:
                print(f'\nSciRepro {task.upper()} | {model.MODEL_NAME}\nWorking directory: {taskstudy}\nResults: {taskdir}\n',flush=True)
                backend=model.make_backend()
                report=run_agent(backend,taskstudy,messages,mode=task)
                metadata['runs'][task]['status']='complete'
            except BaseException as e:
                metadata['runs'][task]['status']='failed'
                metadata['runs'][task]['error']=f'{type(e).__name__}: {e}'
                raise
            finally:
                transcript={'model':model.MODEL_NAME,'backend':model.BACKEND,'mode':task,'messages':messages,'final_report':report}
                (taskdir/'transcript.json').write_text(json.dumps(transcript,indent=2,default=str))
                (taskdir/'report.md').write_text('# '+task.title()+' report\n\n'+(report or '(No report generated)')+'\n')
                metadata['runs'][task]['files_written']=file_manifest(taskstudy)
                (dest/'manifest.json').write_text(json.dumps(metadata,indent=2))
                results[task]=report
        lines=['# SciRepro run summary','',f'Model: `{model.MODEL_NAME}`',f'Mode: `{mode}`','', '## Agent reports (not independently verified)','']
        for task,report in results.items():
            lines.extend([f'### {task.title()}',report,''])
        lines.extend(['## Interpretation','These reports contain agent claims. A successful execution is not proof of scientific correctness. Independent ground truth is needed for quantitative scoring.'])
        (dest/'summary.md').write_text('\n'.join(lines))
    finally:
        (dest/'manifest.json').write_text(json.dumps(metadata,indent=2))
    print(f'\nSciRepro finished. Results saved to: {dest}',flush=True)
    return dest


def main(argv=None):
    p=argparse.ArgumentParser(prog='scirepro',description='Reproduce and audit scientific analyses with an LLM ReAct agent')
    sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run',help='Run a paper/data study with an LLM')
    r.add_argument('--study',required=True,help='Folder containing paper.pdf and data/, optional analysis/ and study.txt')
    r.add_argument('--model',help='Hugging Face model ID or OpenAI-compatible server model ID')
    r.add_argument('--backend',choices=['hf','openai'],help='Inference backend (default hf)')
    r.add_argument('--mode',choices=['auto','reproduce','audit','both'],default='auto')
    r.add_argument('--output',help='Output root directory (default ./outputs); creates timestamped subfolder')
    r.add_argument('--max-turns',type=int)
    r.add_argument('--temperature',type=float)
    args=p.parse_args(argv)
    try:
        return run(args)
    except (ValueError,RuntimeError,FileNotFoundError) as e:
        p.error(str(e))

if __name__=='__main__':
    main()
