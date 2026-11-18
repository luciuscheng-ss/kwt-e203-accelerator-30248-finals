"""Freeze a training-only factor, then optionally evaluate it on validation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
PROJECT = RUN / 'frozen01/project04'
REFERENCE = ROOT / 'artifacts/reference'
JOINT = PROJECT / 'experiments/joint_search_8000_20261111'
CONVERTER = JOINT / 'narrow11'
RECIPE = HERE / 'RECIPE.json'
FROZEN = HERE / 'training_result.json'
LIMIT = 2047
BASE_FACTOR = 1.05

sys.path.insert(0, str(CONVERTER))
from norm_convert import convert, decode


def load(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, payload):
    with path.open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')


def layer_data(partition, layer):
    return tuple(np.load(REFERENCE / partition / f'layer_{layer:02d}_{kind}.npy', mmap_mode='r')
                 for kind in ('input', 'output'))


def operands(inputs, layer, bias):
    cfg = load(CONVERTER / f'training_o13/x11w11_layer_{layer:02d}.json')
    with np.load(JOINT / f'dynamic12/w11_layer_{layer:02d}.npz') as weights:
        qw = weights['qw']
    qx, qw, qb, _, _, sx, _ = convert(inputs, bias, cfg, qw, 4, 13)
    # These bounded integers have an exact double-precision dot product.
    raw = (qx.astype(np.float64) @ qw.astype(np.float64)).astype(np.int64) + qb[:, None, :]
    return raw, cfg, sx


def output_scales(cfg, sx, table, factor):
    bounds = table * factor / BASE_FACTOR
    gain = LIMIT * np.asarray(sx)[..., None] * np.asarray(cfg['weight_scale']) / bounds
    shift = np.clip(np.floor(np.log2(15 / gain)), 0, 38).astype(np.int64)
    mult = np.clip(np.floor(np.ldexp(gain, shift) + 0.5), 1, 15).astype(np.int64)
    half = np.where(shift > 0, 1 << np.maximum(shift - 1, 0), 0)
    return shift, mult, half


def freeze_training(weights):
    setting = load(RECIPE)['factor_candidates_in_hundredths']
    candidates = np.arange(setting['first'], setting['last'] + 1, setting['step'])
    training = REFERENCE / 'calibration'
    manifest = training / 'samples.json'
    assert not FROZEN.exists() and len(load(manifest)) == 600
    worst_by_factor = np.zeros(len(candidates), dtype=np.int64)
    tables = []
    training_hashes = {str(manifest.relative_to(ROOT)): digest(manifest)}
    for layer in range(12):
        inputs, targets = layer_data('calibration', layer)
        assert inputs.shape == (600, 99, 64) and targets.shape == (600, 99, 256)
        table = (BASE_FACTOR * np.max(np.abs(targets), axis=(0, 1))).astype(np.float64)
        tables.append(table.tolist())
        raw, cfg, sx = operands(inputs, layer, weights[f'b{layer:02d}'])
        max_abs = np.max(np.abs(raw), axis=1)
        for index, hundredths in enumerate(candidates):
            shift, mult, half = output_scales(cfg, sx, table, int(hundredths) / 100)
            worst = np.max((max_abs * mult + half) >> shift)
            worst_by_factor[index] = max(worst, worst_by_factor[index])
        for kind in ('input', 'output'):
            path = training / f'layer_{layer:02d}_{kind}.npy'
            training_hashes[str(path.relative_to(ROOT))] = digest(path)
        print('training layer', layer, flush=True)
    passing = candidates[worst_by_factor <= LIMIT]
    if not len(passing):
        raise RuntimeError('No prespecified candidate avoids training clipping')
    selected = int(passing[0]) / 100
    write_new(FROZEN, {'recipe_sha256': digest(RECIPE), 'probe_sha256': digest(Path(__file__)),
                       'v14_converter_sha256': digest(CONVERTER / 'norm_convert.py'),
                       'o12_source_manifest_sha256': digest(PROJECT / 'SOURCE_SHA256.json'),
                       'input_sha256': training_hashes, 'selected_factor': selected,
                       'bound_stored_factor': BASE_FACTOR, 'per_layer_bounds': tables,
                       'training_peak_unsigned_by_candidate': dict(zip(map(str, candidates), map(int, worst_by_factor))),
                       'validation_tensors_opened_during_training': False, 'test_split_accessed': False})
    print('Training-only factor', selected, 'SHA256', digest(FROZEN), flush=True)


def evaluate_validation(weights):
    frozen = load(FROZEN)
    assert frozen['recipe_sha256'] == digest(RECIPE) and frozen['probe_sha256'] == digest(Path(__file__))
    frozen_sha = digest(FROZEN)
    cases = []
    for layer in range(12):
        inputs, targets = layer_data('validation', layer)
        assert len(inputs) == len(targets) == load(RECIPE)['validation_samples_per_layer']
        table = np.array(frozen['per_layer_bounds'][layer])
        energy = squared_error = worst_task = 0.0
        clipped = above = 0
        for start in range(0, len(inputs), 64):
            entry = slice(start, start + 64)
            raw, cfg, sx = operands(inputs[entry], layer, weights[f'b{layer:02d}'])
            shift, mult, half = output_scales(cfg, sx, table, frozen['selected_factor'])
            rounded = (np.abs(raw) * mult[:, None, :] + half[:, None, :]) >> shift[:, None, :]
            clipped += int(np.count_nonzero(rounded > LIMIT))
            codes = np.minimum(rounded, LIMIT) * np.sign(raw)
            expected = targets[entry].astype(np.float64)
            observed = decode(codes, cfg, shift, mult, sx)
            difference = observed - expected
            per_sample_error = np.sum(difference * difference, axis=(1, 2))
            per_sample_energy = np.sum(expected * expected, axis=(1, 2))
            task_errors = np.sqrt(per_sample_error / per_sample_energy)
            assert np.isfinite(task_errors).all()
            energy += float(per_sample_energy.sum())
            squared_error += float(per_sample_error.sum())
            worst_task = max(worst_task, float(task_errors.max()))
            above += int(np.count_nonzero(task_errors > 0.03))
        cases.append({'layer': layer, 'nrmse': (squared_error / energy) ** 0.5,
                      'worst_sample_nrmse': worst_task, 'saturations': clipped, 'samples_over_3pct': above})
        print(cases[-1], flush=True)
    worst_layer = max(c['nrmse'] for c in cases)
    write_new(HERE / 'validation_result.json', {'training_freeze_sha256': frozen_sha,
              'selected_factor': frozen['selected_factor'], 'samples_per_layer': len(inputs),
              'worst_layer_nrmse': worst_layer, 'local_precision_score': 10 - worst_layer / 0.003,
              'worst_sample_nrmse': max(c['worst_sample_nrmse'] for c in cases),
              'samples_over_3pct': sum(c['samples_over_3pct'] for c in cases),
              'cases': cases, 'official_score': None, 'native_hardware_run': False})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('stage', choices=('training', 'validation'))
    args = cli.parse_args()
    with np.load(PROJECT / 'artifacts/reference/weights.npz') as weights:
        if args.stage == 'training':
            freeze_training(weights)
        else:
            evaluate_validation(weights)


if __name__ == '__main__':
    main()
