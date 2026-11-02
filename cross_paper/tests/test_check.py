"""Protocol fixtures test plumbing, not newly reproduced research results."""
import json
from pathlib import Path
import pytest
from cross_paper.check import ROOT, main, replay
from paper_tools.spec_validation_tools import do_load_spec
from paper_tools.state import ReproState

MANIFEST = json.loads((ROOT / 'repositories.json').read_text())
FIXTURES = {
    'uap': 'ROC AUC (model level): 1.0\nROC AUC (view level): 0.9698\n[victim] test accuracy 0.8888\nBest Fooling Rate = 0.7895\n',
    'ipmix': 'Clean\nTest Error 4.0\nMean Corruption Error: 12.0\n',
    'ep_bnp': 'Before prunning\nTest clean accuracy: 93.86\nTest attack success rate: 100.0\nAfter EP prunning\nTest clean accuracy: 93.88\nTest attack success rate: 0.86\nAfter BNP prunning\nTest clean accuracy: 93.6\nTest attack success rate: 1.6\n',
    'ffb': '{"validation": {"val/acc": 99}, "test": {"test/acc": 80.5, "test/dp": 2.5}}',
}


@pytest.mark.parametrize('paper', MANIFEST)
def test_four_paper_protocol_and_saved_verdicts(paper, tmp_path):
    state = ReproState(str(tmp_path / 'state'))
    entry = MANIFEST[paper]
    loaded = do_load_spec(state, str(ROOT / entry['spec']))
    assert loaded['repo'] == entry['repo']
    assert Path(loaded['dockerfile']).is_file()
    results = replay(state, FIXTURES[paper], 'SYNTHETIC TEST FIXTURE')
    assert set(results) == set(state.spec.metrics)
    assert all(r['status'] in ('PASS', 'NO_RECORD') for r in results.values())
    restored = ReproState(state.workspace)
    assert len(restored.data['validations']) == len(results)
    if paper == 'ffb':
        assert results['test_acc']['reproduced'] == 80.5
        assert results['test_dp']['reproduced'] == 2.5
    if paper == 'ep_bnp':
        assert results['ep_asr']['reproduced'] == 0.86
        assert results['bnp_asr']['reproduced'] == 1.6


def test_replay_missing_metric_does_not_validate(tmp_path):
    state = ReproState(str(tmp_path))
    do_load_spec(state, str(ROOT / MANIFEST['uap']['spec']))
    results = replay(state, 'not an experiment output', 'fixture')
    assert all(r['status'] == 'NOT_FOUND' for r in results.values())
    assert not state.data['validations']


def test_cli_reports_replay_separately_and_refuses_reuse(tmp_path):
    log = tmp_path / 'result.json'
    log.write_text(FIXTURES['ffb'])
    output = tmp_path / 'output'
    args = ['ffb', '--log', str(log), '--output', str(output)]
    assert main(args) == 0
    report = json.loads((output / 'report.json').read_text())
    assert report['mode'] == 'replay'
    assert report['new_experiment_executed'] is False
    assert len(report['log_sha256']) == 64
    with pytest.raises(SystemExit):
        main(args)


def test_missing_docker_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr('cross_paper.check.shutil.which', lambda _: None)
    output = tmp_path / 'output'
    assert main(['uap', '--build', '--output', str(output)]) == 1
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'ERROR'
    assert 'Docker CLI unavailable' in report['error']
