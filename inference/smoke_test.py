"""
Synthetic smoke test for the WebSocket policy server.

This is what we hit before involving the real robot. It sends the server
a zeros observation (uint8 frames, zero state, a generic instruction)
and checks that the response arrives, has shape (T, 11), contains no
NaN or Inf, and falls inside a sane per-joint range. It also reports
per-call latency.

Most of the deployment bugs we have seen die here, before anyone gets
near a robot. Checkpoint shape mismatch, missing norm stats, the
attention_mask dtype issue, the tokenizer not pre-cached inside the
container, and so on. A green run of this script is not a guarantee of
robot success, but a red run is a guarantee of robot failure.

Run it with

    python inference/smoke_test.py --url ws://localhost:8000
"""

import argparse
import asyncio
import time

import msgpack
import numpy as np
import websockets


SYNTHETIC_OBS = {
    "head_rgb": np.zeros((480, 640, 3), dtype=np.uint8).tolist(),
    "hand_rgb": np.zeros((480, 640, 3), dtype=np.uint8).tolist(),
    "state": [0.0] * 8,
    "instruction": "pick up the mug",
}


async def run(url: str, n_calls: int):
    print(f"Connecting to {url}")
    async with websockets.connect(url, max_size=20 * 1024 * 1024) as ws:
        latencies = []
        for i in range(n_calls):
            payload = msgpack.packb(SYNTHETIC_OBS, use_bin_type=True)
            t0 = time.perf_counter()
            await ws.send(payload)
            raw = await ws.recv()
            dt_ms = (time.perf_counter() - t0) * 1000

            resp = msgpack.unpackb(raw, raw=False)
            if "error" in resp:
                print(f"  call {i + 1}: server returned an error -> {resp['error']}")
                return 1
            actions = np.asarray(resp["actions"], dtype=np.float32)
            latencies.append(dt_ms)

            ok_shape = actions.ndim == 2 and actions.shape[1] == 11
            ok_finite = bool(np.isfinite(actions).all())
            # |a| < 10 is a loose sanity bound in robot units, not a hard
            # joint limit. Tight bounds need the URDF, which is out of
            # scope for a smoke test.
            ok_range = bool((np.abs(actions) < 10).all())
            verdict = "ok" if (ok_shape and ok_finite and ok_range) else "FAIL"
            print(f"  call {i + 1}: shape={actions.shape}, finite={ok_finite}, "
                  f"|max|={float(np.abs(actions).max()):.3f}, {dt_ms:.1f} ms  [{verdict}]")

            if not ok_shape:
                print(f"    expected shape (T, 11), got {actions.shape}")
            if not ok_finite:
                print(f"    response contained NaN or Inf")
            if not ok_range:
                mins = actions.min(axis=0).tolist()
                maxs = actions.max(axis=0).tolist()
                print(f"    per-joint ranges (min, max) = {list(zip(mins, maxs))}")
            if not (ok_shape and ok_finite and ok_range):
                return 1

        if latencies:
            print(f"\nLatency over {len(latencies)} calls. "
                  f"Mean {np.mean(latencies):.1f} ms, "
                  f"p50 {np.median(latencies):.1f} ms, "
                  f"p95 {np.percentile(latencies, 95):.1f} ms.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Smoke test for the policy server")
    parser.add_argument("--url", type=str, default="ws://localhost:8000")
    parser.add_argument("--n-calls", type=int, default=5)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args.url, args.n_calls)))


if __name__ == "__main__":
    main()
