"""Inspect a paper spec, replay an existing log, or explicitly run Docker.

Default mode only loads configuration. No model/API calls are made.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from paper_tools.extract import extract_value
from paper_tools.spec_validation_tools import do_load_spec, do_validate_metric
from paper_tools.reproduce_tools import do_reproduce_metric
from paper_tools.state import ReproState

ROOT = Path(__file__).resolve().parent


def replay(state, text, origin):
    """Extract available values, then use the team's real validation implementation."""
    results = {}
    for name, metric in state.spec.metrics.items():
        value = extract_value(text, metric)
        if value is None:
            results[name] = {"status": "NOT_FOUND"}
            continue
        state.data['reproduced'][state.key(name)] = {
            "value": value, "origin": str(origin), "mode": "replay"}
        results[name] = do_validate_metric(state, name)
    return results


def main(argv=None):
    manifest = json.loads((ROOT / 'repositories.json').read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paper', choices=manifest)
    parser.add_argument('--output', required=True, type=Path, help='New directory, never reused')
    parser.add_argument('--log', type=Path, help='Existing stdout log, or FFB result.json; no execution')
    parser.add_argument('--repo-path', type=Path, help='Optional local clone: verify committed files and revision')
    parser.add_argument('--build', action='store_true', help='Build candidate paper image using Docker CLI')
    parser.add_argument('--ca-bundle', type=Path, help='Optional approved CA PEM bundle for build HTTPS')
    parser.add_argument('--smoke', action='store_true', help='Run image default framework import check')
    parser.add_argument('--execute', action='store_true', help='Run actual experiment; may train for hours')
    args = parser.parse_args(argv)
    if args.log and (args.build or args.smoke or args.execute):
        parser.error('--log cannot be combined with Docker actions')
    if args.output.exists():
        parser.error('--output must be a new directory (prevents stale cache reuse)')
    args.output.mkdir(parents=True)
    entry = manifest[args.paper]
    report = {'paper': args.paper, 'repo': entry['repo'], 'expected_commit': entry['commit'],
              'mode': 'replay' if args.log else 'execute' if args.execute else 'environment' if args.smoke or args.build else 'spec',
              'status': 'ERROR', 'new_experiment_executed': False}
    try:
        state = ReproState(str(args.output / 'state'))
        report['configuration'] = do_load_spec(state, str(ROOT / entry['spec']))
        if args.repo_path:
            def git(*command):
                return subprocess.check_output(['git', '-C', str(args.repo_path), *command], text=True).strip()
            origin = git('remote', 'get-url', 'origin').removesuffix('.git').rstrip('/')
            if origin != entry['repo']:
                raise ValueError('Repository origin does not match manifest')
            git('cat-file', '-e', entry['commit'] + '^{commit}')
            for path in entry['required_files']:
                git('cat-file', '-e', entry['commit'] + ':' + path)
            report['repository_check'] = 'Pinned commit and required files exist; code was not executed'
        if args.log:
            raw = args.log.read_bytes()
            report['log'] = str(args.log.resolve())
            report['log_sha256'] = hashlib.sha256(raw).hexdigest()
            report['metrics'] = replay(state, raw.decode('utf-8', errors='replace'), args.log.resolve())
        if args.build or args.smoke or args.execute:
            if not shutil.which('docker'):
                raise RuntimeError('Docker CLI unavailable; build and experiment compatibility remain unverified')
            subprocess.run(['docker', 'info'], check=True, capture_output=True, timeout=30)
        if args.build:
            secrets = ['--secret', f'id=pip_ca,src={args.ca_bundle.resolve()}'] if args.ca_bundle else []
            with (args.output / 'build.log').open('w', encoding='utf-8') as log:
                subprocess.run(['docker', 'build', *secrets, '-t', state.spec.image_tag(), '-f',
                                str(ROOT / entry['dockerfile']), str((ROOT / entry['dockerfile']).parent)],
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
            report['build'] = 'OK'
        if args.build or args.smoke or args.execute:
            report['image_id'] = subprocess.check_output(
                ['docker', 'image', 'inspect', '--format', '{{.Id}}', state.spec.image_tag()], text=True, timeout=30).strip()
        if args.smoke:
            with (args.output / 'smoke.log').open('w', encoding='utf-8') as log:
                subprocess.run(['docker', 'run', '--rm', state.spec.image_tag()], check=True,
                               stdout=log, stderr=subprocess.STDOUT, timeout=120)
            report['smoke'] = 'Framework import check passed; dataset and GPU not verified'
        if args.execute:
            missing = [host for host in state.spec.volumes if not Path(state.spec.resolve(host)).is_dir()]
            if missing:
                raise ValueError('Create/populate dataset mounts first: ' + ', '.join(missing))
            report['metrics'] = {}
            for name in state.spec.metrics:
                result = do_reproduce_metric(state, name)
                report['metrics'][name] = {'execution': result}
                if result['status'] == 'OK':
                    report['metrics'][name]['validation'] = do_validate_metric(state, name)
            report['new_experiment_executed'] = bool(state.data['runs'])
        metrics = report.get('metrics', {})
        bad = any(result.get('status') == 'NOT_FOUND' or
                  ('execution' in result and result['execution']['status'] != 'OK') for result in metrics.values())
        report['status'] = 'INCOMPLETE' if bad else 'OK'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'OK' else 1


if __name__ == '__main__':
    raise SystemExit(main())
