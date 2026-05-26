"""Launch NeMo Conformer-CTC training for first-letter anchor.

Usage (on the RunPod pod):

    python scripts/train_first_letter_anchor.py \
        --config training/conformer_ctc_medium_first_letter.yaml \
        --hf-model-repo surindersinghssj/surt-anchor-ctc-first-letter-v1 \
        --hf-push-every 2000

Modes:
  - default: train Conformer-CTC Medium from scratch (PLAN.md spec).
  - --init-from-nemo PATH: load an existing .nemo model and adopt its encoder
    weights into the from-scratch architecture (best-effort; mismatched shapes
    are skipped). Use this to switch to a pretrained-acoustic-encoder ablation.
  - --smoke: cap training at 100 steps for a sanity decode.

Every --hf-push-every steps the script pushes the current best+last NeMo
checkpoint to the HF model repo so progress survives pod loss.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _load_config(config_path: str, overrides: dict):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v, merge=True)
    return cfg


def _attempt_pretrained_encoder_init(model, nemo_path: str) -> None:
    """Copy encoder weights from a pretrained NeMo model. Best-effort.

    Mismatched shapes are skipped with a log line, so this is safe to call
    with checkpoints that have a different decoder vocab / architecture.
    """
    import torch
    import nemo.collections.asr as nemo_asr  # noqa: F401
    from nemo.core import ModelPT

    print(f"[init] loading pretrained encoder from {nemo_path}", flush=True)
    pretrained = ModelPT.restore_from(nemo_path, map_location="cpu", strict=False)
    pre_state = pretrained.encoder.state_dict()
    own_state = model.encoder.state_dict()
    copied = skipped = 0
    for k, v in pre_state.items():
        if k in own_state and own_state[k].shape == v.shape:
            own_state[k] = v.clone()
            copied += 1
        else:
            skipped += 1
    model.encoder.load_state_dict(own_state)
    del pretrained
    torch.cuda.empty_cache()
    print(f"[init] encoder weights copied: {copied} | skipped: {skipped}", flush=True)


def _maybe_push_to_hf(checkpoint_dir: Path, hf_repo: str, step: int):
    """Push the latest .nemo file and any tensorboard logs to HF model repo."""
    from huggingface_hub import HfApi

    nemos = sorted(checkpoint_dir.rglob("*.nemo"))
    if not nemos:
        print(f"[hf-push step={step}] no .nemo files yet, skipping", flush=True)
        return
    latest = max(nemos, key=lambda p: p.stat().st_mtime)
    api = HfApi()
    try:
        api.create_repo(hf_repo, repo_type="model", private=False, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(latest),
            path_in_repo=f"checkpoints/{latest.name}",
            repo_id=hf_repo,
            repo_type="model",
            commit_message=f"checkpoint at step {step}",
        )
        # Also stash a small status file
        status = checkpoint_dir / "STATUS.txt"
        status.write_text(
            f"latest_checkpoint: {latest.name}\nstep: {step}\nts: {time.time()}\n",
            encoding="utf-8",
        )
        api.upload_file(
            path_or_fileobj=str(status),
            path_in_repo="STATUS.txt",
            repo_id=hf_repo,
            repo_type="model",
            commit_message=f"status update at step {step}",
        )
        print(f"[hf-push step={step}] pushed {latest.name} to {hf_repo}", flush=True)
    except Exception as e:
        print(f"[hf-push step={step}] FAILED: {e}", flush=True)


class _HFPushCallback:
    """Lightweight Lightning callback that pushes checkpoints to HF every N steps."""

    def __init__(self, checkpoint_dir: Path, hf_repo: str, every: int):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.hf_repo = hf_repo
        self.every = every
        self._last_push = 0

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        step = int(trainer.global_step)
        if step - self._last_push >= self.every and step > 0:
            self._last_push = step
            _maybe_push_to_hf(self.checkpoint_dir, self.hf_repo, step)

    def on_train_end(self, trainer, pl_module):
        _maybe_push_to_hf(self.checkpoint_dir, self.hf_repo, int(trainer.global_step))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init-from-nemo", default=None,
                    help="Optional .nemo path for pretrained encoder weights.")
    ap.add_argument("--hf-model-repo",
                    default="surindersinghssj/surt-anchor-ctc-first-letter-v1")
    ap.add_argument("--hf-push-every", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true",
                    help="100-step sanity run with verbose decoding.")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Override trainer.max_steps from config.")
    args = ap.parse_args()

    import torch  # noqa: F401
    # NeMo 1.23+ requires the `lightning.pytorch` namespace, not legacy
    # `pytorch_lightning`. Both ship together in lightning>=2.0 but NeMo
    # isinstance-checks against the new one.
    try:
        import lightning.pytorch as pl
    except ImportError:
        import pytorch_lightning as pl
    from nemo.collections.asr.models import EncDecCTCModel
    from nemo.utils.exp_manager import exp_manager

    overrides: dict = {}
    if args.smoke:
        overrides["trainer.max_steps"] = 100
        overrides["trainer.val_check_interval"] = 50
        overrides["trainer.log_every_n_steps"] = 10
        overrides["exp_manager.create_early_stopping_callback"] = False
    if args.max_steps is not None:
        overrides["trainer.max_steps"] = args.max_steps

    cfg = _load_config(args.config, overrides)

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    model = EncDecCTCModel(cfg=cfg.model, trainer=trainer)

    if args.init_from_nemo:
        _attempt_pretrained_encoder_init(model, args.init_from_nemo)

    # HF push callback wraps Lightning's training loop without altering NeMo internals.
    if args.hf_model_repo and not args.smoke:
        callback = _HFPushCallback(
            checkpoint_dir=Path(cfg.exp_manager.exp_dir),
            hf_repo=args.hf_model_repo,
            every=args.hf_push_every,
        )
        trainer.callbacks.append(_LightningCallbackAdapter(callback))

    trainer.fit(model)

    # Final push regardless of smoke mode
    _maybe_push_to_hf(Path(cfg.exp_manager.exp_dir), args.hf_model_repo,
                      int(trainer.global_step))
    return 0


class _LightningCallbackAdapter:
    """Adapter so our duck-typed callback satisfies pl.Callback API."""
    def __init__(self, inner):
        self._inner = inner
    def __getattr__(self, name):
        return getattr(self._inner, name, lambda *a, **kw: None)


if __name__ == "__main__":
    raise SystemExit(main())
