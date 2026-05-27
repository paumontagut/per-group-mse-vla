# Inference skeleton

A self-contained WebSocket policy server for SmolVLA, with the
companion smoke test and the Dockerfile we used while iterating on it.
If `docs/INFERENCE.md` is the contract on paper, this directory is the
contract in code.

Worth saying upfront, this is **not** the workshop evaluator. That one
is the counterpart on the robot side, belongs upstream, and is not
redistributed here (see `docs/CONTEXT.md`). What follows is an
implementation written from scratch, kept small enough to read end to
end.

The directory contains three files. `policy_server.py` loads a SmolVLA
checkpoint, reads the normalization statistics directly from the
checkpoint's safetensors (the choice explained in `docs/CONTEXT.md`),
accepts msgpack-encoded observations on a WebSocket, and returns the
predicted action chunk in the 11-DoF HSR layout. `smoke_test.py` is a
synthetic client that sends a zeros observation, checks the response
shape, finiteness and per-joint ranges, and reports per-call latency.
`Dockerfile` builds a CUDA 12 + Python 3.12 + LeRobot v0.5.1 image
with the SmolVLM processor pre-cached so the server can boot offline.

## Running it locally

In one terminal start the server.

```bash
python inference/policy_server.py \
    --checkpoint /path/to/pretrained_model \
    --host 0.0.0.0 --port 8000
```

In another, smoke-test it.

```bash
python inference/smoke_test.py --url ws://localhost:8000
```

A healthy run prints five `[ok]` lines and a latency summary at the
end. On failure the script exits with status 1 and the line just above
the exit names which assertion fired.

## Running it in Docker

```bash
docker build -f inference/Dockerfile -t smolvla-policy-server .
docker run --rm --runtime=nvidia --gpus all \
    -v /path/to/pretrained_model:/checkpoint \
    -p 8000:8000 \
    smolvla-policy-server
```

The container exposes port 8000 and expects the checkpoint to be
mounted at `/checkpoint`. `--runtime=nvidia` is mandatory. The default
`runc` will not pick the GPU and the server will fall back to CPU, at
which point inference takes seconds and the smoke test times out.

## I/O contract

Documented in full in [`../docs/INFERENCE.md`](../docs/INFERENCE.md).
The request is a dict with `head_rgb`, `hand_rgb`, `state` and
`instruction`. The response is a dict with `actions` of shape
`(T, 11)`. The 11 columns are the HSR action layout, in the order
`arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll, gripper,
head_pan, head_tilt, base_x, base_y, base_theta`.

## Wiring this to a real evaluator

The server is agnostic to who is on the other end of the WebSocket. If
your evaluator already speaks the contract above, point it at the
server's host and port and you are done. If it speaks a different one,
`handle()` in `policy_server.py` is around fifteen lines and is the
only place that needs to change.
