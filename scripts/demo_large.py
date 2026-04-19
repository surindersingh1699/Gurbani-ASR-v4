"""Test different Whisper models on Gurbani with MPS + float16.

Loads one model at a time to fit in 8GB RAM.

Run:
    python scripts/demo_large.py
"""

import time
import gc
import gradio as gr
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np

MODEL_OPTIONS = {
    "Whisper Large-v3 (1.5B, zero-shot)": {
        "model_id": "openai/whisper-large-v3",
        "processor_id": "openai/whisper-large-v3",
    },
    "Surt Small - Kirtan (244M, fine-tuned)": {
        "model_id": "surindersinghssj/surt-small-v1-kirtan",
        "processor_id": "openai/whisper-small",
    },
    "Surt Small - Sehaj Path (244M, fine-tuned)": {
        "model_id": "surindersinghssj/surt-small-v1",
        "processor_id": "openai/whisper-small",
    },
}

if torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
elif torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

print(f"Device: {device}, dtype: {dtype}")

# Only load one model at a time
current = {"name": None, "model": None, "processor": None}


def load_model(name):
    """Load a model, unloading the previous one first."""
    if current["name"] == name:
        return

    # Unload previous
    if current["model"] is not None:
        print(f"Unloading {current['name']}...")
        del current["model"]
        del current["processor"]
        current["model"] = None
        current["processor"] = None
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    cfg = MODEL_OPTIONS[name]
    print(f"Loading {name}...")
    proc = WhisperProcessor.from_pretrained(cfg["processor_id"])
    model = WhisperForConditionalGeneration.from_pretrained(
        cfg["model_id"], torch_dtype=dtype
    )
    model = model.to(device)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {params:.0f}M params loaded on {device}")

    current["name"] = name
    current["model"] = model
    current["processor"] = proc


def _transcribe_array(audio_data, model_choice, use_beam_search, initial_prompt):
    """Transcribe 16kHz float32 audio with chunking."""
    load_model(model_choice)
    model = current["model"]
    proc = current["processor"]

    chunk_len = 30 * 16000
    stride = 29 * 16000

    if len(audio_data) > chunk_len:
        chunks = []
        for start in range(0, len(audio_data), stride):
            chunk = audio_data[start : start + chunk_len]
            if len(chunk) > 16000:
                chunks.append(chunk)
    else:
        chunks = [audio_data]

    gen_kwargs = {
        "no_repeat_ngram_size": 3,
        "condition_on_prev_tokens": False,
        "language": "pa",
        "task": "transcribe",
    }

    if use_beam_search:
        gen_kwargs["num_beams"] = 5

    if initial_prompt and initial_prompt.strip():
        prompt_ids = proc.get_prompt_ids(initial_prompt.strip(), return_tensors="pt")
        gen_kwargs["prompt_ids"] = prompt_ids.to(device)

    all_text = []
    for chunk in chunks:
        input_features = proc(
            chunk, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device=device, dtype=dtype)

        with torch.no_grad():
            predicted_ids = model.generate(input_features, **gen_kwargs)

        text = proc.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        if text and text not in all_text[-1:]:
            all_text.append(text)

    return "\n".join(all_text)


def transcribe_file(file_path, model_choice, use_beam_search, initial_prompt):
    if file_path is None:
        return "No file provided."
    import librosa
    audio_data, _ = librosa.load(file_path, sr=16000)
    duration = len(audio_data) / 16000
    t0 = time.time()
    result = _transcribe_array(audio_data, model_choice, use_beam_search, initial_prompt)
    elapsed = time.time() - t0
    return f"[{duration:.0f}s audio, {elapsed:.1f}s to transcribe]\n\n{result}"


def transcribe_mic(audio, model_choice, use_beam_search, initial_prompt):
    if audio is None:
        return "No audio provided."
    sr, audio_data = audio
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    elif audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    if sr != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
    t0 = time.time()
    result = _transcribe_array(audio_data, model_choice, use_beam_search, initial_prompt)
    elapsed = time.time() - t0
    return f"[{elapsed:.1f}s to transcribe]\n\n{result}"


with gr.Blocks(title="Surt — Gurbani ASR") as demo:
    gr.Markdown("# Surt — Gurbani ASR")
    gr.Markdown(
        "Compare models on Gurbani audio. "
        "Only one model loaded at a time (8GB RAM friendly). "
        "Switching models takes a few seconds."
    )

    with gr.Row():
        model_choice = gr.Radio(
            choices=list(MODEL_OPTIONS.keys()),
            value="Surt Small - Kirtan (244M, fine-tuned)",
            label="Model (switches automatically, previous model unloaded)",
        )

    with gr.Row():
        use_beam = gr.Checkbox(
            value=False,
            label="Beam search (5 beams — slower but better)",
        )
        initial_prompt = gr.Textbox(
            value="",
            label="Initial prompt (Gurmukhi hint)",
            placeholder="e.g. ਨਾਮ ਬਿਨਾ ਨਹੀ ਜੀਵਿਆ ਜਾਇ",
            lines=1,
        )

    output = gr.Textbox(label="Gurmukhi Transcription", lines=8)

    with gr.Tabs():
        with gr.Tab("Upload File"):
            file_input = gr.File(
                label="Upload Audio/Video",
                file_types=[
                    ".wav", ".mp3", ".mp4", ".m4a",
                    ".flac", ".ogg", ".webm",
                ],
            )
            file_btn = gr.Button("Transcribe", variant="primary")
            file_btn.click(
                transcribe_file,
                inputs=[file_input, model_choice, use_beam, initial_prompt],
                outputs=output,
            )

        with gr.Tab("Microphone"):
            mic_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Record",
            )
            mic_btn = gr.Button("Transcribe", variant="primary")
            mic_btn.click(
                transcribe_mic,
                inputs=[mic_input, model_choice, use_beam, initial_prompt],
                outputs=output,
            )

if __name__ == "__main__":
    demo.launch()
