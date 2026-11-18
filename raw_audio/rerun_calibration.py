"""Replay the frozen training procedure against an explicitly supplied reference cache."""
import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys


def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


project = Path(__file__).resolve().parents[1]/'project04'
canonical = project/'calibration/probe.py'
frozen = json.loads((canonical.parent/'training_result.json').read_text())
assert digest(canonical) == frozen['probe_sha256']
cli = argparse.ArgumentParser(description=__doc__)
cli.add_argument('--reference-cache', type=Path, required=True)
cli.add_argument('--out', type=Path, required=True)
args = cli.parse_args()
reference = args.reference_cache.resolve(strict=True)
out = args.out.resolve()
manifest = reference/'calibration/samples.json'
expected = canonical.parent/'training_samples.json'
assert json.loads(manifest.read_text()) == json.loads(expected.read_text())
joint = project/'experiments/joint_search_8000_20261111'
sys.path.insert(0, str(joint/'narrow11'))
spec = importlib.util.spec_from_file_location('frozen_o12_calibration', canonical)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
probe.ROOT, probe.PROJECT, probe.REFERENCE = reference.parent, project, reference
probe.JOINT, probe.CONVERTER = joint, joint/'narrow11'
probe.RECIPE, probe.HERE, probe.FROZEN = canonical.parent/'RECIPE.json', out, out/'training_result.json'
probe.digest = digest
out.mkdir(parents=True)
with probe.np.load(project/'artifacts/reference/weights.npz') as weights:
    probe.freeze_training(weights)
new = json.loads(probe.FROZEN.read_text())
assert new['selected_factor'] == frozen['selected_factor']
probe.np.testing.assert_allclose(new['per_layer_bounds'], frozen['per_layer_bounds'], rtol=1e-6, atol=1e-7)
print('Replayed training-only factor', new['selected_factor'])
