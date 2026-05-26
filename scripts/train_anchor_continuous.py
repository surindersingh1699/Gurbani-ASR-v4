"""v2 continuous training: load Gurbani-pretrained encoder, swap to first-letter CTC head.

Two init paths:
  --init-from-hf <repo>      pulls a .nemo from HF, extracts encoder weights, builds a
                             fresh EncDecCTCModel with our char vocab, copies encoder
                             weights (and preproc cfg) over. Best for v2.0.
  --resume-from-hf <repo>    pulls our own previous .nemo (e.g. v2.0) and restore_from
                             directly. Best for v2.x refreshes.

HF auto-push every --hf-push-every steps.
"""
from __future__ import annotations

import argparse
import os
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


def _download_nemo_from_hf(repo_id: str, filename_hint: str | None = None) -> str:
    """Find and download a .nemo file from an HF repo."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    files = [f for f in api.list_repo_files(repo_id, repo_type="model") if f.endswith(".nemo")]
    if not files:
        raise FileNotFoundError(f"No .nemo files in {repo_id}")
    if filename_hint:
        match = [f for f in files if filename_hint in f]
        if match:
            files = match
    # Prefer files in checkpoints/ subdirectory, then largest
    files.sort(key=lambda f: (0 if "checkpoint" in f else 1, f))
    chosen = files[0]
    print(f"[hf-init] downloading {repo_id}/{chosen}", flush=True)
    return hf_hub_download(repo_id=repo_id, filename=chosen, repo_type="model")


def _build_ctc_model_with_pretrained_encoder(cfg, pretrained_nemo: str):
    """Load a (possibly hybrid) pretrained NeMo model and copy its encoder into a fresh CTC model."""
    import torch
    from nemo.core import ModelPT
    from nemo.collections.asr.models import EncDecCTCModel

    print(f"[init] loading pretrained model from {pretrained_nemo}", flush=True)
    pretrained = ModelPT.restore_from(pretrained_nemo, map_location="cpu", strict=False)
    pre_state = pretrained.encoder.state_dict()

    print(f"[init] building fresh CTC model (will overwrite encoder)", flush=True)
    model = EncDecCTCModel(cfg=cfg.model)

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
    print(f"[init] encoder weights copied={copied}  skipped={skipped}  "
          f"(skipped >0 is normal if pretrained had extra heads)", flush=True)
    return model


def _maybe_push_to_hf(checkpoint_dir: Path, hf_repo: str, step: int):
    from huggingface_hub import HfApi
    nemos = sorted(checkpoint_dir.rglob("*.nemo"))
    if not nemos:
        return
    latest = max(nemos, key=lambda p: p.stat().st_mtime)
    api = HfApi()
    try:
        api.create_repo(hf_repo, repo_type="model", private=False, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(latest),
            path_in_repo=f"checkpoints/{latest.name}",
            repo_id=hf_repo, repo_type="model",
            commit_message=f"checkpoint at step {step}",
        )
        status = checkpoint_dir / "STATUS.txt"
        status.write_text(
            f"latest_checkpoint: {latest.name}\nstep: {step}\nts: {time.time()}\n",
            encoding="utf-8",
        )
        api.upload_file(path_or_fileobj=str(status), path_in_repo="STATUS.txt",
                        repo_id=hf_repo, repo_type="model",
                        commit_message=f"status step {step}")
        print(f"[hf-push step={step}] pushed {latest.name} to {hf_repo}", flush=True)
    except Exception as e:
        print(f"[hf-push step={step}] FAILED: {e}", flush=True)


class _HFPushCallback:
    def __init__(self, checkpoint_dir, hf_repo, every):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.hf_repo = hf_repo
        self.every = every
        self._last_push = 0

    def on_train_batch_end(self, trainer, pl_module, *a, **kw):
        step = int(trainer.global_step)
        if step - self._last_push >= self.every and step > 0:
            self._last_push = step
            _maybe_push_to_hf(self.checkpoint_dir, self.hf_repo, step)

    def on_train_end(self, trainer, pl_module):
        _maybe_push_to_hf(self.checkpoint_dir, self.hf_repo, int(trainer.global_step))


class _LightningCallbackAdapter:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name, lambda *a, **kw: None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init-from-hf", default=None,
                    help="HF model repo to take encoder weights from (e.g. indicconformer-pa-v3-kirtan)")
    ap.add_argument("--resume-from-hf", default=None,
                    help="HF model repo to restore_from directly (own previous .nemo)")
    ap.add_argument("--hf-model-repo",
                    default="surindersinghssj/surt-anchor-ctc-large-v2")
    ap.add_argument("--hf-push-every", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    try:
        import lightning.pytorch as pl
    except ImportError:
        import pytorch_lightning as pl
    from nemo.collections.asr.models import EncDecCTCModel
    from nemo.utils.exp_manager import exp_manager

    overrides = {}
    if args.smoke:
        overrides["trainer.max_steps"] = 50
        overrides["trainer.val_check_interval"] = 25
        overrides["trainer.log_every_n_steps"] = 5
        overrides["exp_manager.create_early_stopping_callback"] = False
    if args.max_steps is not None:
        overrides["trainer.max_steps"] = args.max_steps

    cfg = _load_config(args.config, overrides)

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    if args.resume_from_hf:
        nemo_path = _download_nemo_from_hf(args.resume_from_hf)
        model = EncDecCTCModel.restore_from(nemo_path, map_location="cpu")
        model._trainer = trainer
    elif args.init_from_hf:
        nemo_path = _download_nemo_from_hf(args.init_from_hf)
        model = _build_ctc_model_with_pretrained_encoder(cfg, nemo_path)
        model._trainer = trainer
    else:
        model = EncDecCTCModel(cfg=cfg.model, trainer=trainer)

    if args.hf_model_repo and not args.smoke:
        callback = _HFPushCallback(
            checkpoint_dir=Path(cfg.exp_manager.exp_dir),
            hf_repo=args.hf_model_repo, every=args.hf_push_every)
        trainer.callbacks.append(_LightningCallbackAdapter(callback))

    trainer.fit(model)
    _maybe_push_to_hf(Path(cfg.exp_manager.exp_dir), args.hf_model_repo,
                      int(trainer.global_step))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
