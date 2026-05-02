# eval

`bench_one.py` runs the live STTM pipeline against a single 16 kHz mono wav and
emits a prediction JSON in the public-benchmark schema.

Scoring lives in the public benchmark, not here:
<https://github.com/karanbirsingh/live-gurbani-captioning-benchmark-v1>

## Run

```bash
source .venv/bin/activate
python eval/bench_one.py --wav path/to/<video_id>_16k.wav --out preds/<video_id>.json
```

Drives `python -m apps.transcribe.app --bench-wav <wav>`, parses
`[sttm] pushed audio_t=… banidb_gurmukhi=…` events from stdout, and writes the
pred JSON.

Flags:

```
--wav <path>       input wav (16 kHz mono), required
--out <path>       output pred JSON, required
--video-id <id>    video id stamped into the pred (default: wav stem minus _16k)
--log <path>       bench stdout log (default: <out>.log)
--no-rerun         reuse an existing log instead of re-running the pipeline
```

## Score

Clone the public benchmark, drop the pred next to a matching ground-truth file,
and run its scorer:

```bash
git clone https://github.com/karanbirsingh/live-gurbani-captioning-benchmark-v1
cd live-gurbani-captioning-benchmark-v1
python eval.py --pred /abs/path/to/<video_id>.json --gt test/<video_id>.json --collar 2
```

`bench_one.py` prints this command on success.

## Iterate

1. tweak something in `apps/transcribe/`
2. `python eval/bench_one.py --wav … --out …`
3. score against the public benchmark
4. compare the headline against the previous run

## Baseline reference

`eval/baseline/IZOsmkdmmcg.json` is the output of `bench_one.py` for the first
benchmark clip, scored at **39.8% frame accuracy** against the public
benchmark (collar=2). `eval/baseline/visualize.html` is the rendered tile
(GT / Pred / Diff strips) from `visualize.py`. Both are committed so changes
to the pipeline can be diffed against a known reference without re-running.

To regenerate:

```bash
python eval/bench_one.py \
  --wav /path/to/IZOsmkdmmcg_16k.wav \
  --out eval/baseline/IZOsmkdmmcg.json
# then, in a clone of the public benchmark:
python visualize.py \
  --pred /abs/path/to/eval/baseline/ \
  --gt test/ \
  --out /abs/path/to/eval/baseline/visualize.html \
  --collar 2
```

