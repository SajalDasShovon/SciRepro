from scirepro.study import prepare
from scirepro.cli import file_manifest

def test_autogenerate_reproduction(tmp_path):
    study=tmp_path/'input'; study.mkdir()
    (study/'paper.txt').write_text('A trial reports 10 participants and a treatment effect.')
    (study/'data').mkdir(); (study/'data'/'sample.csv').write_text('x\n1\n')
    out=tmp_path/'output'; out.mkdir()
    info=prepare(study,'reproduce',out)
    assert info['instruction_source']=='generated'
    assert 'sample.csv' in (out/'study.txt').read_text()
    assert '10 participants' in (out/'paper_extracted.txt').read_text()

def test_preserves_provided_instructions(tmp_path):
    (tmp_path/'study.txt').write_text('Specific controlled instructions')
    assert prepare(tmp_path,'audit',tmp_path)['instruction_source']=='provided'

def test_manifest(tmp_path):
    (tmp_path/'a.csv').write_text('x\n')
    assert len(file_manifest(tmp_path)[0]['sha256'])==64
