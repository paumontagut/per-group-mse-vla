"""
WebSocket policy server for SmolVLA on the Toyota HSR.

This is the skeleton we used to put a SmolVLA checkpoint behind the kind
of WebSocket contract that mobile-manipulation evaluators tend to expect.
It is NOT the workshop evaluator (that one belongs upstream and is not
redistributed here, see docs/CONTEXT.md). It is a small self-contained
server that you can read end to end in a sitting, deploy in a Docker
container, and bend to your own evaluator if you need to.

The endpoint accepts a JSON-ish payload (msgpack, see below) with the
following keys.

    head_rgb     uint8 (480, 640, 3)   head camera frame
    hand_rgb     uint8 (480, 640, 3)   hand camera frame
    state        float32 (8,)          5 arm + 1 gripper + 2 head, robot units
    instruction  str                   natural-language prompt

It responds with

    actions      float32 (T, 11)       T future actions, 11-DoF layout
                                       arm(5) + gripper(1) + head(2) + base(3)

The action chunk is what the policy emits in one forward pass, around 50
steps at the configured action horizon. The client typically plays a
prefix and asks for the next chunk before the prefix runs out.

We deliberately avoid the full LeRobot inference pipeline at runtime,
because keeping a six-step preprocessing pipeline in sync across LeRobot
versions was fragile in practice. Instead we load the normalization
statistics directly from the safetensors of the checkpoint and apply
them in this file. The trade-off is documented in docs/CONTEXT.md.

Run it with

    python inference/policy_server.py \\
        --checkpoint /path/to/pretrained_model \\
        --host 0.0.0.0 --port 8000

and verify with `inference/smoke_test.py` before connecting any robot.
"""

import argparse
import asyncio
import logging
import time
from pathlib import Path

import msgpack
import numpy as np
import torch
import websockets
from safetensors import safe_open

LOG = logging.getLogger("policy_server")

# Mirrors eval/per_group_mse.py and matches the dataset layout.
ACTION_DIM = 11
STATE_DIM = 8

# Tokenizer's attention_mask comes back as int64 on this code path.
# SmolVLA's eager attention expects bool. See docs/CONTEXT.md ("attention_mask dtype").
ATTENTION_MASK_DTYPE = torch.bool


def load_norm_stats(checkpoint_dir: Path):
    """Load the normalization statistics that the checkpoint shipped with.

    The exact filename varies a bit between LeRobot versions, so we look
    for the usual suspects rather than hard-coding one. Returns a dict
    with float32 numpy arrays.
    """
    candidates = [
        checkpoint_dir / "norm_stats.safetensors",
        checkpoint_dir / "stats.safetensors",
    ]
    safetensors_path = next((p for p in candidates if p.exists()), None)
    if safetensors_path is None:
        raise FileNotFoundError(
            f"No norm_stats.safetensors or stats.safetensors under {checkpoint_dir}. "
            "If the checkpoint stores stats elsewhere, point at it explicitly."
        )

    stats = {}
    with safe_open(safetensors_path, framework="pt") as f:
        for key in f.keys():
            stats[key] = f.get_tensor(key).float().cpu().numpy()
    LOG.info("Loaded %d norm-stats tensors from %s", len(stats), safetensors_path.name)
    return stats


def normalize_state(state: np.ndarray, stats: dict) -> np.ndarray:
    """Apply mean/std normalization to the 8-D state vector.

    Falls back to quantile normalization (q01, q99) if mean/std are not
    in the stats dict, which is the case for some LeRobot versions.
    """
    if "state_mean" in stats and "state_std" in stats:
        return (state - stats["state_mean"]) / (stats["state_std"] + 1e-8)
    if "state_q01" in stats and "state_q99" in stats:
        q01, q99 = stats["state_q01"], stats["state_q99"]
        return 2.0 * (state - q01) / (q99 - q01 + 1e-8) - 1.0
    raise KeyError("Norm stats for the state vector not found in the checkpoint.")


def unnormalize_action(action: np.ndarray, stats: dict) -> np.ndarray:
    """Undo the same transform on the predicted action chunk."""
    if "action_mean" in stats and "action_std" in stats:
        return action * stats["action_std"] + stats["action_mean"]
    if "action_q01" in stats and "action_q99" in stats:
        q01, q99 = stats["action_q01"], stats["action_q99"]
        return ((action + 1.0) / 2.0) * (q99 - q01) + q01
    raise KeyError("Norm stats for the action vector not found in the checkpoint.")


def _img_to_tensor(img: np.ndarray, device: str) -> torch.Tensor:
    t = torch.from_numpy(img).to(device).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0)


class Server:
    def __init__(self, checkpoint_dir: Path, device: str = "cuda"):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from transformers import AutoTokenizer

        self.device = device
        self.policy = SmolVLAPolicy.from_pretrained(str(checkpoint_dir))
        self.policy.eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
        self.stats = load_norm_stats(checkpoint_dir)
        LOG.info("Policy server ready on %s", device)

    @torch.no_grad()
    def infer(self, obs: dict) -> np.ndarray:
        """One forward pass. Returns (T, 11) float32 in robot units."""
        head_rgb = np.asarray(obs["head_rgb"], dtype=np.uint8)
        hand_rgb = np.asarray(obs["hand_rgb"], dtype=np.uint8)
        state = np.asarray(obs["state"], dtype=np.float32)
        instruction = str(obs.get("instruction", ""))
        if state.shape != (STATE_DIM,):
            raise ValueError(f"state must be shape ({STATE_DIM},), got {state.shape}")

        state_norm = normalize_state(state, self.stats).astype(np.float32)
        enc = self.tokenizer(instruction, return_tensors="pt",
                             padding=True, truncation=True)

        batch = {
            "observation.images.camera1": _img_to_tensor(head_rgb, self.device),
            "observation.images.camera2": _img_to_tensor(hand_rgb, self.device),
            "observation.state": torch.from_numpy(state_norm).float()
                                      .to(self.device).unsqueeze(0),
            "input_ids": enc["input_ids"].to(self.device),
            # attention_mask must be bool, not the int64 the tokenizer
            # returns, see docs/CONTEXT.md.
            "attention_mask": enc["attention_mask"].to(self.device).to(ATTENTION_MASK_DTYPE),
        }

        # predict_action_chunk returns (1, T, action_dim). Slicing to
        # ACTION_DIM is defensive in case the policy was trained with
        # padded actions.
        pred = self.policy.predict_action_chunk(batch)
        chunk_norm = pred[0, :, :ACTION_DIM].cpu().numpy()
        return unnormalize_action(chunk_norm, self.stats).astype(np.float32)


async def handle(ws, server: Server):
    LOG.info("Client connected from %s", ws.remote_address)
    async for message in ws:
        t0 = time.perf_counter()
        try:
            obs = msgpack.unpackb(message, raw=False)
            actions = server.infer(obs)
            response = {"actions": actions.tolist()}
            latency_ms = (time.perf_counter() - t0) * 1000
            LOG.info("Inference OK, chunk shape %s, %.1f ms", actions.shape, latency_ms)
            await ws.send(msgpack.packb(response, use_bin_type=True))
        except Exception as e:
            LOG.exception("Inference failed")
            await ws.send(msgpack.packb({"error": str(e)}, use_bin_type=True))


async def serve(checkpoint_dir: Path, host: str, port: int, device: str):
    server = Server(checkpoint_dir, device=device)
    # 20 MB cap on incoming payloads. A pair of 480x640x3 uint8 frames is
    # ~1.8 MB raw, msgpack-packed lists are bigger but well below 20 MB.
    async with websockets.serve(lambda ws: handle(ws, server), host, port,
                                max_size=20 * 1024 * 1024):
        LOG.info("Listening on ws://%s:%d", host, port)
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description="SmolVLA WebSocket policy server")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the pretrained_model directory of a SmolVLA checkpoint")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    )
    asyncio.run(serve(Path(args.checkpoint), args.host, args.port, args.device))


if __name__ == "__main__":
    main()
