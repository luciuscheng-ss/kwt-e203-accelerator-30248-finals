"""Generate 12 complete fixed12 SoC cases from official-format WAV input.

Run in the pinned reference environment. The SoC loader sees only source.hex;
raw.hex, codes.hex and matrix.hex belong solely to the verification oracle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import wave


HERE = Path(__file__).resolve().parent
if (HERE.parent / "project04").is_dir():
    # Standalone delivery layout.
    ROOT = HERE.parent
    PROJECT = ROOT / "project04"
    PORTABLE = ROOT / "numeric"
else:
    # Development workspace layout.
    ROOT = HERE.parents[2]
    BASE = HERE.parent.parent / "conservative_bus1_mode0_raw16_20261113"
    PROJECT = BASE / "project04"
    PORTABLE = BASE / "portable04"
SAMPLE_RATE = 16000
ROWS, COLUMNS = 99, 256
SOURCE_WORDS = 8168
OUTPUT_WORDS = ROWS * COLUMNS * 12 // 32


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_hex(stream, values, bits):
    width = (bits + 3) // 4
    mask = (1 << bits) - 1
    stream.writelines(f"{int(value) & mask:0{width}x}\n" for value in values)


def tile_order(matrix):
    # RTL retires all rows of one 16-column group before the next group.
    return matrix.reshape(ROWS, COLUMNS // 16, 16).transpose(1, 0, 2).ravel()


def verify_wav(path):
    with wave.open(str(path), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (
            SAMPLE_RATE, 1, 2,
        ):
            raise ValueError(f"Expected 16 kHz mono 16-bit PCM WAV: {path}")
        if source.getnframes() == 0:
            raise ValueError(f"Empty WAV: {path}")


def manifest_inputs(args):
    values = args.audio
    if len(values) not in (1, 12):
        raise ValueError("Supply one WAV for all layers or one input for each of 12 layers")
    if args.validation_root and args.purpose != "validation":
        raise ValueError("--validation-root applies only to validation runs")
    if args.purpose == "validation" and not args.validation_root:
        raise ValueError("Validation runs require --validation-root")
    validation_paths = set()
    if args.validation_root:
        root = args.validation_root.resolve()
        validation_paths = set((root / "validation_list.txt").read_text().splitlines())
    files = []
    for item in values:
        if item == "silence":
            files.append(None)
            continue
        path = Path(item).resolve(strict=True)
        verify_wav(path)
        if args.validation_root:
            relative = path.relative_to(root).as_posix()
            if relative not in validation_paths:
                raise ValueError(f"Audio is outside the supplied validation list: {relative}")
        files.append(path)
    return files * 12 if len(files) == 1 else files


def extract_features(inputs, reference, np):
    # An empty data_dir tells the official processor to prepare its audio graph
    # without enumerating a corpus or downloading data.
    flags, session, processor, model, _, probe = reference.build_reference("")
    try:
        features = {}
        for item in dict.fromkeys(inputs):
            if item is None:
                audio = np.zeros((1, SAMPLE_RATE), dtype=np.float32)
            else:
                audio = np.asarray(
                    processor.get_features_for_wav(str(item), flags, session), dtype=np.float32
                )
            if audio.shape != (1, SAMPLE_RATE) or not np.all(np.isfinite(audio)):
                raise ValueError("Invalid WAV decoder output")
            output = probe.predict_on_batch(audio)
            np.testing.assert_allclose(output[-2], output[-1], rtol=1e-5, atol=1e-5)
            features[item] = [output[i][0] for i in range(1, 25)]
        return features, model._checkpoint_match_count
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", nargs="+", required=True, metavar="WAV_OR_SILENCE")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--portable-dir", type=Path, default=PORTABLE)
    parser.add_argument("--reference-root", type=Path, default=ROOT)
    parser.add_argument("--purpose", choices=["validation", "synthetic", "evaluation"], required=True)
    parser.add_argument("--validation-root", type=Path)
    args = parser.parse_args()

    inputs = manifest_inputs(args)
    project = args.project.resolve(strict=True)
    sys.path.insert(0, str(args.reference_root.resolve() / "scripts"))
    sys.path.insert(0, str(args.portable_dir.resolve()))
    sys.path.insert(0, str(project / "experiments/shared_bus_write_allocate_20261111"))
    import numpy as np
    import reference_pipeline as reference
    from numeric_reference_native51 import FREEZE, frozen_configs, reference_task
    from encode_task_native51 import encode_task, decode_output_words

    frozen_configs()
    features, checkpoint_tensors = extract_features(inputs, reference, np)
    args.out.mkdir(parents=True)
    tensors = args.out / "tensors"
    vectors = args.out / "vectors"
    tensors.mkdir()
    vectors.mkdir()
    names = ("source.hex", "raw.hex", "codes.hex", "matrix.hex", "word_counts.hex")
    records = []
    from contextlib import ExitStack
    with ExitStack() as stack:
        streams = {name: stack.enter_context((vectors / name).open("w")) for name in names}
        for layer, item in enumerate(inputs):
            x, floating = features[item][2 * layer:2 * layer + 2]
            input_file = tensors / f"layer_{layer:02d}_input.npy"
            float_file = tensors / f"layer_{layer:02d}_float_output.npy"
            np.save(input_file, x)
            np.save(float_file, floating)
            # The input packet is generated from X and frozen config before
            # computing or reading any expected GEMM result.
            source, metadata = encode_task(x, layer)
            expected = reference_task(x, layer)
            np.testing.assert_array_equal(source, expected["source_words"])
            full_matrix = expected["matrix_words"]
            decoded = np.asarray(decode_output_words(full_matrix.tolist(), metadata)).reshape(ROWS, COLUMNS)
            np.testing.assert_allclose(decoded, expected["decoded"], rtol=1e-13, atol=0)
            normalized_error = np.linalg.norm(decoded - floating) / max(
                np.linalg.norm(floating.astype(np.float64)), np.sqrt(floating.size) * 1e-8
            )
            write_hex(streams["source.hex"], source, 32)
            write_hex(streams["raw.hex"], tile_order(expected["raw"]), 33)
            write_hex(streams["codes.hex"], tile_order(expected["codes"]), 14)
            write_hex(streams["matrix.hex"], full_matrix, 32)
            write_hex(streams["word_counts.hex"], [OUTPUT_WORDS], 32)
            records.append({
                "layer": layer,
                "input": item.name if item else "silence",
                "input_wav_sha256": sha256(item) if item else None,
                "layer_input_sha256": sha256(input_file),
                "float_output_sha256": sha256(float_file),
                "source_words": len(source),
                "output_words": len(full_matrix),
                "software_reference_nrmse": float(normalized_error),
            })
    (vectors / "dual_params.vh").write_text(
        f"`define TEST_CASES {len(records)}\n`define MATRIX_TOTAL {len(records) * OUTPUT_WORDS}\n"
    )
    weights = reference.SOURCE / "models_data_v2_12_labels/kwt1"
    evidence = {
        "purpose": args.purpose,
        "official_score": None,
        "native_soc_run": False,
        "calibration_or_numeric_parameter_selection": False,
        "numeric_freeze_sha256": FREEZE,
        "checkpoint_tensors_verified_bit_exact": checkpoint_tensors,
        "reference_checkpoint_sha256": {
            file.name: sha256(file) for file in sorted(weights.glob("best_weights.*"))
        },
        "reference_probe_sha256": sha256(Path(reference.__file__)),
        "cases": records,
        "vector_sha256": {name: sha256(vectors / name) for name in (*names, "dual_params.vh")},
    }
    (vectors / "prepared_vectors.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Prepared {len(records)} cases within {vectors}")


if __name__ == "__main__":
    main()
