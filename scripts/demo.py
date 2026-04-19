"""Gradio demo for Surt — Gurbani ASR.

Run:
    pip install gradio transformers torch librosa
    python scripts/demo.py

Opens a browser tab where you can record or upload audio
and get Gurmukhi transcription back. Supports both Sehaj Path
and Kirtan models.
"""

import gradio as gr
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np

MODELS = {
    "Sehaj Path (WER 14.88%)": "surindersinghssj/surt-small-v1",
    "Kirtan (WER 32.65%)": "surindersinghssj/surt-small-v1-kirtan",
}

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

# Load processor once (same for both models)
print("Loading processor...")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# Load both models
loaded_models = {}
for name, model_id in MODELS.items():
    print(f"Loading {name} from {model_id}...")
    model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    loaded_models[name] = model

print(f"Both models loaded on {device} with {dtype}")


def _transcribe_array(audio_data, model_choice):
    """Transcribe a 16kHz float32 numpy array to Gurmukhi text.

    Chunks long audio into 30-second segments with 1-second overlap.
    """
    model = loaded_models[model_choice]
    chunk_len = 30 * 16000  # 30 seconds
    stride = 29 * 16000     # 1 second overlap

    # Split into chunks for audio longer than 30s
    if len(audio_data) > chunk_len:
        chunks = []
        for start in range(0, len(audio_data), stride):
            chunk = audio_data[start : start + chunk_len]
            if len(chunk) > 16000:  # skip chunks < 1 second
                chunks.append(chunk)
    else:
        chunks = [audio_data]

    all_text = []
    for chunk in chunks:
        input_features = processor(
            chunk, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device=device, dtype=dtype)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                no_repeat_ngram_size=3,
                condition_on_prev_tokens=False,
            )

        text = processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        if text and text not in all_text[-1:]:  # skip exact duplicates
            all_text.append(text)

    return "\n".join(all_text)


def transcribe_file(file_path, model_choice):
    """Transcribe from a file path (supports mp4, mp3, m4a, wav, etc.)."""
    if file_path is None:
        return "No file provided."
    import librosa
    audio_data, sr = librosa.load(file_path, sr=16000)
    return _transcribe_array(audio_data, model_choice)


def transcribe_mic(audio, model_choice):
    """Transcribe from microphone numpy input."""
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
    return _transcribe_array(audio_data, model_choice)


model_selector = gr.Radio(
    choices=list(MODELS.keys()),
    value="Sehaj Path (WER 14.88%)",
    label="Model",
)

with gr.Blocks(title="Surt — Gurbani ASR") as demo:
    gr.Markdown("# Surt — Gurbani ASR")
    gr.Markdown(
        "Automatic speech recognition for Gurbani in Gurmukhi script. "
        "Upload audio files (mp3, mp4, wav, m4a, flac) or record from microphone."
    )

    model_choice = gr.Radio(
        choices=list(MODELS.keys()),
        value="Sehaj Path (WER 14.88%)",
        label="Model",
    )
    output = gr.Textbox(label="Gurmukhi Transcription", lines=3)

    with gr.Tabs():
        with gr.Tab("Upload File"):
            file_input = gr.File(
                label="Upload Audio/Video",
                file_types=[".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".webm"],
            )
            file_btn = gr.Button("Transcribe")
            file_btn.click(transcribe_file, inputs=[file_input, model_choice], outputs=output)

        with gr.Tab("Microphone"):
            mic_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Record Audio",
            )
            mic_btn = gr.Button("Transcribe")
            mic_btn.click(transcribe_mic, inputs=[mic_input, model_choice], outputs=output)

if __name__ == "__main__":
    demo.launch()
