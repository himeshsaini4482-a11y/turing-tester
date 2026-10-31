# Adding your own model

Turing Tester recognises Qwen3-TTS, MOSS-TTS-Nano and Orpheus folders from their `config.json`. Any other
model needs one small file in its folder. Pick the one that matches the model:

| Your model | Add | Benchmark | SFT | DPO |
|---|---|---|---|---|
| a language model that outputs audio-codec tokens (one token stream, Hugging Face `AutoModelForCausalLM`) | `turing-tester.yaml` | ✅ | ✅ LoRA | ✅ |
| anything else (VITS, Kokoro, F5-TTS, a diffusion model, your own code) | `synthesize.py` | ✅ | with a `train.py` | ❌ |

If the model needs extra Python packages, add a `requirements.txt` to the folder. Turing Tester then
builds a separate environment for it (PyTorch, numpy, soundfile and your requirements) the first time.

## `synthesize.py`

```python
# models/my-tts/synthesize.py
import numpy as np

CLONES_VOICE = True          # False if the model cannot clone (ref_audio is then None)

def load(device):            # called once; device is "cuda:0", "mps" or "cpu"
    global model
    model = ...              # load your weights from this folder

def synthesize(text, ref_audio, ref_text):
    """Speak `text` in the voice of the clip `ref_audio` (a wav path), whose transcript is `ref_text`.
    Return (mono float32 numpy array, sample rate)."""
    wav = model.tts(text, speaker_wav=ref_audio)
    return np.asarray(wav, dtype=np.float32), 24000
```

The folder is put on `sys.path`, so `synthesize.py` can import its own helper modules. Turing Tester seeds
`torch` and `numpy` before each sentence, so reruns sample the same way.

### Optional `train.py` (makes `turing-tester sft` work)

```python
def train(records_jsonl, output_dir, device):
    """records_jsonl: one {"audio", "text", "ref_audio", "speaker", "duration"} per line.
    Write a complete model folder (with synthesize.py) to output_dir."""
```

Turing Tester prepares the data (your recordings, the teacher's filtered clips or EARS chunks), calls
`train()`, and then benchmarks and checks the folder you wrote like any other checkpoint.

## `turing-tester.yaml` (LLM-style models)

For a model that is one causal language model over text tokens plus audio-codec tokens, the way
Orpheus is. The folder must load with `AutoModelForCausalLM.from_pretrained` and
`AutoTokenizer.from_pretrained`. This is Orpheus written as a `turing-tester.yaml`:

```yaml
type: token-lm
codec: snac_24khz                 # built in, or the name of a .py file in this folder (below)
audio_token_offset: 128266        # snac_24khz only: token id of SNAC code 0 in the first of 7 slots
stop_token: 128258                # the token that ends the audio
pad_token: 128258                 # optional, default stop_token
prompt:                           # what the model sees, in order
  - [128259]                      #   a list = literal token ids
  - ref_text                      #   the reference clip's transcript (tokenized with the folder's tokenizer)
  - [128009, 128260, 128261, 128257]
  - ref_audio                     #   the reference clip as codec tokens
  - [128258, 128262]
  - [128259]
  - text                          #   the sentence to speak
  - [128009, 128260, 128261, 128257]
sampling: {temperature: 0.6, top_p: 0.95, repetition_penalty: 1.1}
max_new_tokens: 2400
```

Leave `ref_audio` and `ref_text` out of the prompt if the model cannot clone voices.

### A codec file

If the model does not use Orpheus' SNAC layout, write the codec as a Python file in the folder and
name it in `codec:`:

```python
# models/my-tts/my_codec.py
sample_rate = 24000          # of decode()'s output
frame_rate = 12.5            # audio frames per second
tokens_per_frame = 8         # LM tokens per audio frame

def load(device):
    return MyCodec.from_pretrained(...).to(device)   # anything; passed back to encode/decode

def encode(state, wav_path):
    """A wav file -> the LM token ids for it (offsets already added)."""

def decode(state, token_ids):
    """LM token ids (stop token removed) -> mono float32 numpy array. Raise ValueError if invalid."""

def is_audio_token(token_id):   # optional: which ids are audio (for counting frames)
    return token_id >= 100000
```

### How the LLM-style model is trained

- **SFT** (`sft/train_lm.py`): LoRA (rank 16, alpha 32, attention and MLP projections), learning
  rate 1e-4, 32 sequences per step, loss on the audio tokens only. The input is exactly the
  inference prompt. These settings were not tuned in our experiments, which trained MOSS-TTS-Nano.
- **DPO** (`dpo/adapters/tokenlm.py`): the same LoRA and the same DPO method and settings as our
  MOSS-TTS-Nano runs. Before round 1, `dpo/gates.py` checks on the GPU that the training
  log-probabilities match the sampler's. If a gate fails, `runs/dpo/<name>/gates.json` says which.

## Checking a new model quickly

```bash
./turing-tester benchmark --model models/my-tts --limit 5 --judges btrm
```

This runs five sentences end to end. Listen to `runs/my-tts_limit5/tts/*.wav` before starting a full run.
