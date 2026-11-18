"""Study discovery and transparent automatic task preparation."""
from pathlib import Path
import re

CODE_SUFFIXES = {'.py', '.r', '.R', '.ipynb', '.do', '.m', '.sas', '.jl'}
DATA_SUFFIXES = {'.csv', '.tsv', '.parquet', '.xlsx', '.json', '.feather', '.dta'}


def extract_paper(study: Path) -> tuple[str, str]:
    papers = sorted([p for p in study.rglob('*') if p.is_file() and p.suffix.lower() in {'.pdf', '.txt', '.md'} and ('paper' in p.stem.lower() or 'article' in p.stem.lower() or 'manuscript' in p.stem.lower())])
    if not papers:
        return '', ''
    p = papers[0]
    if p.suffix.lower() == '.pdf':
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError('PDF detected. Install PDF support: pip install "scirepro-agent[pdf]"') from e
        text = '\n'.join(page.extract_text() or '' for page in PdfReader(str(p)).pages)
        if len(text.strip()) < 100:
            raise RuntimeError('Paper PDF has little extractable text (possibly scanned). Supply a text-readable PDF or paper.txt.')
    else:
        text = p.read_text(encoding='utf-8', errors='replace')
    return str(p.relative_to(study)), text


def discover(study: Path) -> dict:
    files = [p for p in study.rglob('*') if p.is_file() and not any(s.startswith('.') for s in p.relative_to(study).parts)]
    code = sorted(str(p.relative_to(study)) for p in files if p.suffix in CODE_SUFFIXES)
    data = sorted(str(p.relative_to(study)) for p in files if p.suffix.lower() in DATA_SUFFIXES)
    paper, text = extract_paper(study)
    return {'paper':paper, 'paper_text':text, 'code':code, 'data':data, 'existing_instructions':(study/'study.txt').is_file()}


def prepare(study: Path, mode: str, generated: Path) -> dict:
    info = discover(study)
    if info['existing_instructions']:
        if info['paper'] and info['paper'].lower().endswith('.pdf'):
            (generated/'paper_extracted.txt').write_text(info['paper_text'])
        return {**{k:v for k,v in info.items() if k!='paper_text'}, 'instruction_source':'provided'}
    if not info['paper']:
        raise ValueError('Supply paper.pdf or paper.txt (or a custom study.txt) in the study folder.')
    if not info['data']:
        raise ValueError('No recognized dataset found. Add data files (e.g. .csv) or provide study.txt for a custom layout.')
    if mode == 'audit' and not info['code']:
        raise ValueError('Audit mode requires original analysis code.')
    task = 'Reproduce numerical findings and analyses reported in the paper.' if mode=='reproduce' else 'Audit the original analysis code against the paper and datasets.'
    instructions = f'''Scientific study: {info['paper']}

Objective: {task}

Available datasets: {', '.join(info['data'])}
Original analysis code: {', '.join(info['code']) if info['code'] else '(none supplied)'}

Read the paper and identify reported results relevant to the available data.
Identify the precise cohort, variable definitions, unit of analysis, exclusions,
statistical procedure, and results before computing. Reconstruct the analysis
independently using executed Python. Compare computed results to the paper,
report discrepancies and explain potential causes. Cite filenames and executed
outputs. Do not fabricate results. Mark findings not verifiable from the
provided data as unresolved. {'Inspect the original code for methodological errors, test hypotheses and report supported corrections.' if mode=='audit' else ''}

Paper text is available in paper_extracted.txt. The extracted text is a reading aid;
check against the original PDF for tables, equations, and extraction errors.
'''
    (generated/'study.txt').write_text(instructions)
    (generated/'paper_extracted.txt').write_text(info['paper_text'])
    return {**{k:v for k,v in info.items() if k!='paper_text'}, 'instruction_source':'generated'}
