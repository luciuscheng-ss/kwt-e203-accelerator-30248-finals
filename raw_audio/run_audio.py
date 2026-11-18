"""Stage freshly generated raw-audio vectors and run the supplied SoC regression.

This adapter keeps the frozen runner and project read-only. The staged runner
uses the same proprietary platform, injected firmware and bus monitors, with a
12-case audio batch instead of the bundled 14-case development fixture.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "project04"
if not BASE.is_dir():
    BASE = HERE.parent.parent / "conservative_bus1_mode0_raw16_20261113" / "project04"
RUNNER_SHA = "f241274f3290436d114f4ac014456868b21a4231b42c981d1a73d82c6e1780ae"
FREEZE_SHA = "79d3238de73ac3849ea1bdd91e7725ad9d15461164843c5d8b9560c0e41785cd"
HEX_COUNTS = {
    "source.hex": 8168 * 12,
    "raw.hex": 25344 * 12,
    "codes.hex": 25344 * 12,
    "matrix.hex": 9504 * 12,
    "word_counts.hex": 12,
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_vectors(vectors):
    evidence = json.loads((vectors / "prepared_vectors.json").read_text())
    if evidence["numeric_freeze_sha256"] != FREEZE_SHA:
        raise ValueError("Wrong numeric freeze")
    if [case["layer"] for case in evidence["cases"]] != list(range(12)):
        raise ValueError("Expected exactly one case per layer, layers 0 through 11")
    names = set(HEX_COUNTS) | {"dual_params.vh"}
    if set(evidence["vector_sha256"]) != names:
        raise ValueError("Expected exactly the fixed12 vector files")
    for name in sorted(names):
        actual = vectors / name
        if sha(actual) != evidence["vector_sha256"][name]:
            raise ValueError("Vector checksum mismatch: " + name)
        if name in HEX_COUNTS and len(actual.read_text().splitlines()) != HEX_COUNTS[name]:
            raise ValueError("Vector count mismatch: " + name)
    if (vectors / "word_counts.hex").read_text().split() != ["00002520"] * 12:
        raise ValueError("Each result must contain 9504 complete 32-bit words")
    if (vectors / "dual_params.vh").read_text().split() != [
        "`define", "TEST_CASES", "12", "`define", "MATRIX_TOTAL", "114048",
    ]:
        raise ValueError("Vector include does not describe 12 full matrices")
    return evidence


def replace_once(text, original, replacement):
    if text.count(original) != 1:
        raise ValueError("Unexpected supplied verification implementation")
    return text.replace(original, replacement)


def stage_project(project, vectors, output):
    frozen = project / "verification/run_mode0.py"
    if sha(frozen) != RUNNER_SHA:
        raise ValueError("Unrecognized frozen SoC runner")
    destination = output / "staged_project"
    verification = destination / "verification"
    verification.mkdir(parents=True)
    for directory in ("hardware", "firmware"):
        (destination / directory).symlink_to(project / directory, target_is_directory=True)
    for name in ("cpu_bindings.py", "loader_cpu_tb.sv"):
        (verification / name).symlink_to(project / "verification" / name)
    (verification / "vectors").symlink_to(vectors, target_is_directory=True)
    runner = frozen.read_text()
    runner = replace_once(runner, "range(1 if a.smoke else 14)", "range(1 if a.smoke else 12)")
    # Record the exact input payload with every RTL and mapped result.
    runner = replace_once(
        runner, "'official_score':None}",
        "'official_score':None,'audio_matrix_input_manifest_sha256':"
        "sha(V/'vectors'/'prepared_vectors.json')}",
    )
    generated = verification / "run_mode0.py"
    generated.write_text(runner)
    readback = (project / "verification/result_readback82.py").read_text()
    # Export only native SRAM Q values from the existing post-task check.
    # This code runs after the final write acknowledgment and VCD capture.
    readback = replace_once(
        readback,
        "'integer j;reg [31:0] address_offset;','begin',",
        "'integer j;integer audio_output_fd;reg [31:0] address_offset;','begin',"
        "\n         ' audio_output_fd=$fopen(\"observed_matrix.hex\",\"w\");',"
        "\n         ' if(!audio_output_fd)$fatal(1,\"Cannot record hardware SRAM output\");',",
    )
    readback = replace_once(
        readback,
        "'  physical_pin_readback_words=physical_pin_readback_words+1;',",
        "'  $fdisplay(audio_output_fd,\"%08h\",rb_q[rb_bank]);',"
        "\n         '  physical_pin_readback_words=physical_pin_readback_words+1;',",
    )
    readback = replace_once(
        readback, "         'end','endtask','']",
        "         ' $fclose(audio_output_fd);', 'end','endtask','']",
    )
    oracle = verification / "result_readback82.py"
    oracle.write_text(readback)
    compile(runner, str(generated), "exec")
    compile(readback, str(oracle), "exec")
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=BASE)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--soc", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--activity", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    if not args.stage_only and not (args.prepared and args.soc):
        parser.error("Native execution requires --prepared and --soc")
    project = args.project.resolve(strict=True)
    vectors = args.vectors.resolve(strict=True)
    out = args.out.resolve()
    check_vectors(vectors)
    out.mkdir(parents=True)
    runner = stage_project(project, vectors, out)
    record = {
        "native_soc_run": False,
        "vector_manifest_sha256": sha(vectors / "prepared_vectors.json"),
        "original_runner_sha256": RUNNER_SHA,
        "staged_runner_sha256": sha(runner),
        "staged_readback_generator_sha256": sha(runner.with_name("result_readback82.py")),
    }
    (out / "stage_result.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.stage_only:
        return
    regression = out / "regression"
    command = [sys.executable, str(runner), "--prepared", str(args.prepared.resolve(strict=True)),
               "--soc", str(args.soc.resolve(strict=True)), "--out", str(regression)]
    if args.mapping:
        command += ["--mapping", str(args.mapping.resolve(strict=True))]
    if args.smoke:
        command.append("--smoke")
    if args.activity:
        command.append("--activity")
    subprocess.run(command, check=True)
    measurement = json.loads((regression / "result.json").read_text())
    if not measurement["pass"]:
        raise RuntimeError("Supplied SoC verification did not pass")
    matrices = {}
    for item in measurement["cases"]:
        label = "case%d_gap%d_rsp%d" % (item["case"], item["command_gap"], item["response_gap"])
        observed = regression / label / "observed_matrix.hex"
        if len(observed.read_text().splitlines()) != 9504:
            raise RuntimeError("Incomplete physical SRAM output: " + label)
        matrices[label] = sha(observed)
    record.update(native_soc_run=True, scope=measurement["implementation"],
                  regression_sha256=sha(regression / "result.json"), observed_matrices=matrices)
    (out / "stage_result.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
