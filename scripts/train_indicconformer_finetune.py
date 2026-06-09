"""Fine-tune the AI4Bharat IndicConformer-pa hybrid model on our Gurbani manifests.

The base model's tokenizer is a multilingual AGGREGATE tokenizer whose per-language
BPE files live INSIDE the .nemo (nemo:<hash> refs). So we cannot build the model
from a standalone cfg.model.tokenizer (that raises "tokenizer.type cannot be None").
Instead we RESTORE the base model (which resolves the in-.nemo tokenizer), then
reconfigure train/val data + augmentation + optimizer, and train.

Usage: python train_indicconformer_finetune.py <config.yaml>
"""
import sys
import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict
import nemo.collections.asr as nemo_asr
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


def main(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    logging.info(f"Restoring base model from {cfg.init_from_nemo_model}")
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
        cfg.init_from_nemo_model, trainer=trainer, map_location="cpu")

    # reconfigure data (this applies train_ds.augmentor: speed/noise/RIR too)
    model.setup_training_data(cfg.model.train_ds)
    model.setup_validation_data(cfg.model.validation_ds)

    # persist new config + apply spec augment + optim
    with open_dict(model.cfg):
        model.cfg.train_ds = cfg.model.train_ds
        model.cfg.validation_ds = cfg.model.validation_ds
        model.cfg.optim = cfg.model.optim
        if "spec_augment" in cfg.model:
            model.cfg.spec_augment = cfg.model.spec_augment
    if "spec_augment" in cfg.model:
        model.spec_augmentation = model.from_config_dict(cfg.model.spec_augment)
    model.setup_optimization(cfg.model.optim)

    # use CER for the logged WER metric if requested (kirtan: CER is primary)
    if cfg.model.get("use_cer", False):
        try:
            model.wer.use_cer = True
            model.ctc_wer.use_cer = True
        except Exception as e:
            logging.warning(f"could not set use_cer: {e}")

    trainer.fit(model)

    # always export a .nemo of the final/best model
    out = f"{cfg.exp_manager.exp_dir}/{cfg.name}.nemo"
    model.save_to(out)
    logging.info(f"saved fine-tuned model -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
