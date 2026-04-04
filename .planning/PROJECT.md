# Surt — Phase 1: Fine-tune Whisper Small on Gurbani

## What This Is

Fine-tune OpenAI's Whisper Small model on 100 hours of sehaj path (sequential reading of Sri Guru Granth Sahib) audio from HuggingFace to produce `surt_small_v1` — a Gurmukhi-specialized ASR model. This is Phase 1 of the larger Surt project, a real-time shabad identification system for live kirtan. Training runs on RunPod A40 via SSH, with checkpoints pushed incrementally to HuggingFace Hub to preserve progress across restarts.

## Core Value

The model must accurately transcribe Gurbani audio in Gurmukhi script — this is the foundation that every downstream phase (forced alignment, curriculum training, distillation) depends on.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Fine-tune Whisper Small (encoder + decoder) on 100h sehaj path FLAC dataset from HuggingFace
- [ ] Stream dataset directly from HuggingFace (no local disk copy needed)
- [ ] Use discriminative learning rates (encoder 5e-5, decoder 1e-4)
- [ ] Apply audio augmentation (Gaussian noise, room simulation, time stretch, pitch shift)
- [ ] Use Mool Mantar as `initial_prompt` during generation for free accuracy boost
- [ ] Push best checkpoint to HuggingFace Hub every 300 steps to preserve progress
- [ ] Support resuming training from last HuggingFace checkpoint if session dies
- [ ] Auto-detect GPU and adjust batch size accordingly
- [ ] Produce `surt_small_v1` model on HuggingFace Hub

### Out of Scope

- Evaluation scripts and exit criteria checking — manual evaluation for now
- Phases 2-5 (forced alignment, curriculum training, distillation, quantization) — future milestones
- Phase 0 data prep and SGGS FAISS index — separate work
- Local inference pipeline — not needed until Phase 5
- Web UI or API serving — not part of training

## Context

- **Base model:** `openai/whisper-small` — Whisper saw very little Punjabi (~0.06% of training data), so both encoder and decoder need significant adaptation
- **Dataset:** 100h sehaj path FLAC on HuggingFace, line-level labeled with Gurmukhi text. Exact dataset path TBD (user will provide)
- **Language:** Punjabi (Gurmukhi script) — `language="pa"`, `task="transcribe"`
- **Model lineage:** This produces `surt_small_v1`, which becomes the starting point for Phase 2 (forced alignment) and Phase 3 (curriculum fine-tuning)
- **Training plan reference:** `surt_training_plan.md` in project root contains the full 6-phase Surt pipeline spec

## Constraints

- **Compute:** RunPod A40 (48GB VRAM) via SSH — Claude Code manages training unattended
- **Cost target:** ~$18-27 (A40 at ~$1.50/hr for 12-18h depending on efficiency)
- **Checkpoint safety:** Must push to HuggingFace incrementally — RunPod sessions can be interrupted
- **No Colab dependency:** Everything must run as Python scripts (no notebook-only patterns)
- **Dataset streaming:** Must stream from HuggingFace, not download to RunPod disk

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Full fine-tuning (not LoRA/adapter) | Whisper has minimal Punjabi exposure; encoder needs deep adaptation to Gurbani acoustics | — Pending |
| RunPod A40 over Colab Pro | Stable SSH session, no session limits, Claude Code can manage unattended | — Pending |
| Discriminative LR (encoder 5e-5, decoder 1e-4) | Encoder knows general audio; decoder learning Gurmukhi nearly from scratch needs higher LR | — Pending |
| HuggingFace Hub as checkpoint store | Preserves progress across session restarts, accessible from any machine | — Pending |

---
*Last updated: 2026-04-04 after initialization*
