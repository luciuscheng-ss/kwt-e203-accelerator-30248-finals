"""Extract official KWT-1 FFN data; never train or alter the supplied weights.

Run using the pinned Python 3.8 environment. All model layers and data indexing
come from the supplied upstream source; this module adds inference probes only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "2026芯片设计赛道比赛文件/KWT-1参考模型"
sys.path.insert(0, str(SOURCE))

import numpy as np
import tensorflow.compat.v1 as tf
from kws_streaming.data import input_data
from kws_streaming.models import utils  # initialize upstream's circular model registry first
from kws_streaming.models import kws_transformer
from kws_streaming.models.transformer_utils import KWSTransformer
from kws_streaming.layers.speech_features import SpeechFeatures
from kws_streaming.train import base_parser


def verify_checkpoint_tensors(sess, model, tr, checkpoint):
    reader = tf.train.load_checkpoint(checkpoint)
    tensors = []
    def layer_pairs(layer, path, names):
        values = layer.get_weights()
        if len(values) != len(names):
            raise RuntimeError("Unexpected weights in " + path)
        for name, value in zip(names, values):
            tensors.append((path + "/" + name, value))
    prefix = "layer_with_weights-0"
    tensors.extend([(prefix + "/pos_emb", sess.run(tr.pos_emb)),
                    (prefix + "/class_emb", sess.run(tr.class_emb))])
    layer_pairs(tr.patch_proj, prefix + "/patch_proj", ["kernel", "bias"])
    for i, block in enumerate(tr.enc_layers):
        path = prefix + "/enc_layers/%d" % i
        for name in ["query_dense", "key_dense", "value_dense"]:
            layer_pairs(getattr(block.att, name), path + "/att/" + name, ["kernel"])
        layer_pairs(block.att.combine_heads, path + "/att/combine_heads", ["kernel", "bias"])
        for layer, name in [(block.ffn.layers[0], "layer_with_weights-0"),
                            (block.ffn.layers[2], "layer_with_weights-1")]:
            layer_pairs(layer, path + "/ffn/" + name, ["kernel", "bias"])
        for name in ["layernorm1", "layernorm2"]:
            layer_pairs(getattr(block, name), path + "/" + name, ["gamma", "beta"])
    layer_pairs(model.layers[-1].layers[0], "layer_with_weights-1/layer_with_weights-0", ["kernel", "bias"])
    for path, value in tensors:
        if not np.array_equal(value, reader.get_tensor(path + "/.ATTRIBUTES/VARIABLE_VALUE")):
            raise RuntimeError("Checkpoint restore mismatch: " + path)
    return len(tensors)


def build_reference(data_dir):
    parser = base_parser.base_parser()
    kws_transformer.model_parameters(parser)
    flags = parser.parse_args([])
    saved = json.loads((SOURCE / "models_data_v2_12_labels/kwt1/flags.json").read_text())
    for key, value in saved.items():
        setattr(flags, key, value)
    flags.data_dir = str(data_dir)
    flags.data_url = ""
    flags.train_dir = str(SOURCE / "models_data_v2_12_labels/kwt1")
    flags.batch_size = None
    flags.train = 0
    flags.training = False
    flags.use_spec_augment = 0
    flags.use_spec_cutout = 0
    flags.time_shift_ms = flags.sp_time_shift_ms = 0.0
    flags.resample = flags.sp_resample = 0.0
    flags.pick_deterministically = 1
    config = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=2,
                            device_count={"GPU": 0})
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)
    tf.keras.backend.set_learning_phase(0)
    tf.set_random_seed(59185)
    np.random.seed(59185)
    processor = input_data.AudioProcessor(flags)
    model = kws_transformer.model(flags)
    status = model.load_weights(str(Path(flags.train_dir) / "best_weights"))
    status.assert_existing_objects_matched()
    status.expect_partial()  # optimizer state in training checkpoint is unused
    tr = next(layer for layer in model.layers if isinstance(layer, KWSTransformer))
    model._checkpoint_match_count = verify_checkpoint_tensors(sess, model, tr, str(Path(flags.train_dir) / "best_weights"))
    sf = next(layer for layer in model.layers if isinstance(layer, SpeechFeatures))
    audio = model.inputs[0]
    features = sf(audio)
    net = tr.patch_proj(features)
    cls = tf.broadcast_to(tr.class_emb, [tf.shape(net)[0], 1, 64])
    net = tf.concat([cls, net], axis=1) + tr.pos_emb
    outputs = [net]
    for block in tr.enc_layers:
        attn, _ = block.att(net)
        normalized = block.layernorm1(net + attn)
        linear = block.ffn.layers[0](normalized)
        net = block.layernorm2(normalized + block.ffn.layers[2](block.ffn.layers[1](linear)))
        outputs.extend([normalized, linear])
    logits = model.layers[-1](net[:, 0])
    outputs.extend([logits, model.outputs[0]])
    probe = tf.keras.Model(audio, outputs)
    return flags, sess, processor, model, tr, probe


def balanced_subset(processor, partition, limit):
    records = processor.data_index[partition]
    if not limit or limit >= len(records):
        return records
    buckets = [[] for _ in range(12)]
    for record in records:
        buckets[processor.word_to_index[record["label"]]].append(record)
    result = []
    while len(result) < limit and any(buckets):
        for bucket in buckets:
            if bucket and len(result) < limit:
                result.append(bucket.pop(0))
    return result


def records_from_manifest(processor, flags, partition, manifest, data_dir):
    """Preserve membership/order across nondeterministic parallel gfile.Glob calls."""
    records=[]
    data_dir=data_dir.resolve()
    for item in manifest:
        path=(data_dir/item["file"]).resolve()
        path.relative_to(data_dir)
        if not path.is_file():
            raise RuntimeError("Frozen sample is missing: "+str(path))
        word=input_data.SILENCE_LABEL if item["silence"] else path.parent.name
        if processor.word_to_index[word]!=item["label"]:
            raise RuntimeError("Frozen sample label mismatch")
        if not item["silence"] and input_data.which_set(str(path),flags.validation_percentage,flags.testing_percentage)!=partition:
            raise RuntimeError("Frozen sample partition mismatch")
        records.append({"file":str(path),"label":word})
    return records


def extract(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    flags, sess, processor, model, tr, probe = build_reference(args.data)
    weights = {}
    for layer_id, block in enumerate(tr.enc_layers):
        w, b = block.ffn.layers[0].get_weights()
        if w.shape != (64, 256) or b.shape != (256,):
            raise RuntimeError("Unexpected FFN shape")
        weights["w%02d" % layer_id] = w
        weights["b%02d" % layer_id] = b
    np.savez(out / "weights.npz", **weights)
    metadata = {"tensorflow": tf.__version__, "numpy": np.__version__,
                "checkpoint_tensors_verified_bit_exact": model._checkpoint_match_count,
                "weight_sha256": hashlib.sha256((Path(flags.train_dir) / "best_weights.data-00000-of-00001").read_bytes()).hexdigest(),
                "inference_mode": True, "training_or_finetuning": False, "partitions": {}}
    for partition, name, limit in [("training", "calibration", args.calibration),
                                    ("validation", "validation", args.validation)]:
        previous_manifest=out/name/"samples.json"
        if previous_manifest.exists():
            previous=json.loads(previous_manifest.read_text())
        else:
            previous=[]
        if previous and (not limit or len(previous)==limit):
            records=records_from_manifest(processor,flags,partition,previous,args.data)
        else:
            records = balanced_subset(processor, partition, limit)
        processor.data_index[partition] = records
        n = len(records)
        destination = out / name
        destination.mkdir(exist_ok=True)
        arrays = []
        for layer_id in range(12):
            arrays.append((
                np.lib.format.open_memmap(str(destination / ("layer_%02d_input.npy" % layer_id)), mode="w+", dtype="float32", shape=(n, 99, 64)),
                np.lib.format.open_memmap(str(destination / ("layer_%02d_output.npy" % layer_id)), mode="w+", dtype="float32", shape=(n, 99, 256))))
        encoded = np.lib.format.open_memmap(str(destination / "encoder_input.npy"), mode="w+", dtype="float32", shape=(n, 99, 64))
        manifest = [{"label": int(processor.word_to_index[r["label"]]),
                     "file": str(Path(r["file"]).relative_to(args.data)),
                     "silence": r["label"] == input_data.SILENCE_LABEL} for r in records]
        (destination / "samples.json").write_text(json.dumps(manifest, indent=2) + "\n")
        correct = 0
        max_probe_difference = 0.0
        for start in range(0, n, args.batch):
            batch = min(args.batch, n - start)
            audio, labels = processor.get_data(batch, start, flags, 0., 0., 0,
                                               partition, 0., 0., sess)
            data = probe.predict_on_batch(audio.astype("float32"))
            error = float(np.max(np.abs(data[-1] - data[-2])))
            max_probe_difference = max(max_probe_difference, error)
            if not np.allclose(data[-1], data[-2], rtol=1e-5, atol=1e-5):
                raise RuntimeError("Probe changed official inference: %r" % error)
            stop = start + batch
            encoded[start:stop] = data[0]
            for layer_id, (inputs, outputs) in enumerate(arrays):
                inputs[start:stop] = data[1 + 2 * layer_id]
                outputs[start:stop] = data[2 + 2 * layer_id]
            correct += int(np.sum(np.argmax(data[-1], axis=1) == labels))
            if start % (args.batch * 10) == 0 or stop == n:
                print("%s %d/%d, reference Top-1 %.3f%%, probe maxdiff %.3g" %
                      (name, stop, n, 100 * correct / stop, max_probe_difference), flush=True)
        for inputs, outputs in arrays:
            inputs.flush(); outputs.flush()
        encoded.flush()
        metadata["partitions"][name] = {"source_partition": partition, "samples": n,
                                            "correct": correct, "top1": correct / n,
                                            "probe_max_abs_difference": max_probe_difference}
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/speech_commands_v0.02")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/reference")
    parser.add_argument("--calibration", type=int, default=600)
    parser.add_argument("--validation", type=int, default=0, help="0 means the entire validation split")
    parser.add_argument("--batch", type=int, default=32)
    extract(parser.parse_args())
