"""Generate evaluation vectors from WAV, then invoke the supplied SoC runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wav', nargs='+', required=True)
    parser.add_argument('--purpose', required=True, choices=('validation', 'evaluation'))
    parser.add_argument('--validation-root', type=Path)
    parser.add_argument('--python-reference', type=Path, required=True)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--soc', type=Path)
    parser.add_argument('--mapping', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage-only', action='store_true')
    args = parser.parse_args()
    if not args.stage_only and not (args.prepared and args.soc):
        parser.error('SoC evaluation requires --prepared and --soc')

    package = HERE.parent
    project = package / 'project04'
    source_manifest = project / 'SOURCE_SHA256.json'
    fmt = json.loads((project/'FIXED12_FORMAT.json').read_text())
    for relative, expected in json.loads(source_manifest.read_text()).items():
        if digest(project / relative) != expected:
            raise ValueError('Source manifest mismatch: ' + relative)
    out = args.out.resolve()
    out.mkdir(parents=True)
    wav = [name if name == 'silence' else str(Path(name).resolve(strict=True)) for name in args.wav]
    args.python_reference.resolve(strict=True)
    generate = [args.python_reference.absolute(), '-B', HERE/'prepare_audio.py',
                '--audio'] + wav + ['--purpose', args.purpose, '--out', out/'vectors_from_wav']
    if args.validation_root:
        generate += ['--validation-root', args.validation_root.resolve(strict=True)]
    environment = dict(os.environ, PYTHONNOUSERSITE='1', CUDA_VISIBLE_DEVICES='-1',
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    environment.pop('PYTHONPATH', None)
    environment.pop('PYTHONHOME', None)
    with (out/'prepare_audio.log').open('x') as log:
        subprocess.run(list(map(str, generate)), cwd=str(package), env=environment,
                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
    vectors = out/'vectors_from_wav/vectors'
    get = lambda path: json.loads(path.read_text())
    packet = get(vectors/'prepared_vectors.json')
    assert packet['numeric_freeze_sha256'] == fmt['numeric_freeze_sha256']
    assert packet['checkpoint_tensors_verified_bit_exact'] == 162 and len(packet['cases']) == 12

    stages = ('staged',) if args.stage_only else (('source', 'mapped') if args.mapping else ('source',))
    checks = {}
    for name in stages:
        stage_output = out/name
        run = [sys.executable, '-B', HERE/'run_audio.py', '--vectors', vectors, '--out', stage_output]
        if args.stage_only:
            run.append('--stage-only')
        else:
            run += ['--prepared', args.prepared.resolve(strict=True), '--soc', args.soc.resolve(strict=True)]
        if name == 'mapped':
            run += ['--mapping', args.mapping.resolve(strict=True)]
        with (out/(name+'.log')).open('x') as log:
            subprocess.run(list(map(str, run)), cwd=str(package), env=environment,
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        stage_record = stage_output/'stage_result.json'
        proof = get(stage_record)
        assert proof['vector_manifest_sha256'] == digest(vectors/'prepared_vectors.json')
        assert proof['native_soc_run'] == (not args.stage_only)
        if not args.stage_only:
            assert len(proof['observed_matrices']) == 14
            regression = get(stage_output/'regression/result.json')
            assert regression['numeric_freeze_sha256'] == fmt['numeric_freeze_sha256']
            assert regression['firmware_sha256'] == digest(project/'firmware/cpu_copy_driver64.hex')
            if name == 'mapped':
                assert regression['mapped_sha256'] == get(args.mapping)['mapped_sha256']
        checks[name] = digest(stage_record)
    record = {'pass': True, 'purpose': args.purpose, 'native_soc_run': not args.stage_only,
              'source_manifest_sha256': digest(source_manifest), 'numeric_sha256': fmt['numeric_freeze_sha256'],
              'vector_manifest_sha256': digest(vectors/'prepared_vectors.json'),
              'checkpoint_tensors': 162, 'ordered_input_layers': len(packet['cases']), 'stage_sha256': checks,
              'interpreter_sha256': digest(args.python_reference.resolve(strict=True))}
    (out/'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
