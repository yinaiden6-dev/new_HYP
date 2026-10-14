"""Keep PP-OCRv6 weights unchanged and reduce CTC probabilities on the GPU.

Only argmax and maximum across the character dimension are appended to the
official recognition graph. This avoids copying N x T x 18710 probabilities
to CPU; the official PaddleX CTC decoder consumes the two small N x T arrays.
"""

import hashlib
import json
from pathlib import Path
import shutil

import paddle

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights/ppocrv6_medium"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    paddle.enable_static()
    paddle.set_device("cpu")
    executor = paddle.static.Executor(paddle.CPUPlace())
    source = WEIGHTS / "PP-OCRv6_medium_rec"
    program, names, outputs = paddle.static.load_inference_model(str(source / "inference"), executor)
    with paddle.static.program_guard(program):
        indices = paddle.argmax(outputs[0], axis=-1)
        scores = paddle.max(outputs[0], axis=-1)
    feed_vars = []
    for name in names:
        for op in program.global_block().ops:
            if op.name() in ("pd_op.data", "pd_op.feed") and op.attrs().get("name") == name:
                feed_vars.append(op.result(0))
                break
    assert len(feed_vars) == len(names)
    target = WEIGHTS / "PP-OCRv6_medium_rec_gpu_ctc"
    target.mkdir(exist_ok=True)
    paddle.static.save_inference_model(str(target / "inference"), feed_vars,
                                       [indices, scores], executor, program=program)
    shutil.copy2(source / "inference.yml", target / "inference.yml")
    assert digest(source / "inference.pdiparams") == digest(target / "inference.pdiparams")
    provenance_path = WEIGHTS / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["runtime_optimization"] = dict(
        kind="GPU CTC argmax/max; unchanged FP32 model parameters and official CTC decoding",
        source_model="PaddlePaddle/PP-OCRv6_medium_rec",
        source_graph_sha256=digest(source / "inference.json"),
        source_and_exported_parameters_sha256=digest(source / "inference.pdiparams"),
        export_script="scripts/export_ppocrv6_ctc.py",
        export_script_sha256=digest(__file__),
        files=[dict(path=str(p.relative_to(WEIGHTS)), size=p.stat().st_size, sha256=digest(p))
               for p in sorted(target.iterdir()) if p.is_file()],
    )
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance["runtime_optimization"], indent=2))


if __name__ == "__main__":
    main()
