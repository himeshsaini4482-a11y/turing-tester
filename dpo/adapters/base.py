"""The adapter contract: everything the model-agnostic DPO core needs from one TTS model.

The core (prompts.py, rollout.py, score_btrm.py, pairs.py, train.py, gates.py, rounds.py) never
touches a model directly. It calls an Adapter, so supporting another TTS means writing one file in
dpo/adapters/ and one entry in hparams.json "models". An adapter must pass gates.py (D1-D5)
before train.py will use it.

Objects the core passes around (all CPU tensors, saved with torch.save in the round's code shards):
  prompt      whatever the model conditions on for one (text, voice): a LongTensor, built once
              per prompt and replicated K times. Must be exactly what generate() sampled from.
  completion  the sampled output tokens of one candidate, *including* the stop token when the
              sample stopped (ended=True). sequence_logps scores exactly these tokens.
Terms: "frame" = the model's audio time step (MOSS 12.5/s, Orpheus SNAC ~11.7/s);
frame_rate converts frames to seconds for the duration gate and per-frame k_hat.
"""
from dataclasses import dataclass


@dataclass
class Voice:
    speaker: str
    wav: str               # absolute path of the reference clip
    transcript: str        # its text (Orpheus and Qwen3-TTS condition on it; MOSS voice_clone does not)


@dataclass
class Sample:
    completion: object     # LongTensor, see above
    ended: bool            # the model emitted its stop token before max_new
    n_frames: int          # audio frames in the completion (0 = no audio)


class Adapter:
    """Subclasses set these and implement every method below."""

    name = "base"
    frame_rate = None          # audio frames per second
    decoded_sample_rate = None  # sample rate of decode() output
    deployment_sampling = {}   # the benchmark generator's decoding settings (benchmark/generate_*.py)
    on_policy_sampling = {}    # temperature 1, no truncation, no penalties: exact pi_theta samples

    def __init__(self, init_dir, policy_dir, device, cfg, for_training=False, repo_dir=None):
        """Load the policy from policy_dir (pi_theta_r) for rollouts, or for training
        (for_training=True: fp32 master weights, trainable params, plus whatever reference_logps
        needs). init_dir is pi_ref (hparams models.<name>.policy_init); cfg is models.<name>."""
        raise NotImplementedError

    # ---- rollout ---------------------------------------------------------------------------
    def prepare_text(self, text):
        """Normalised text as the benchmark generator would feed it, or None to drop the prompt
        (e.g. over the model's single-chunk token limit). Returns (text_or_None, n_text_tokens)."""
        raise NotImplementedError

    def build_prompt(self, text, voice):
        """prompt LongTensor for (prepared text, Voice). May cache per-voice work (codec codes)."""
        raise NotImplementedError

    def generate(self, prompts, sampling, max_new, seed):
        """Sample one completion per prompt (the core passes each prompt K times), batched.
        Returns [Sample]. sampling is a kwargs dict (deployment_sampling or on_policy_sampling)."""
        raise NotImplementedError

    def max_new_for_seconds(self, seconds):
        """max_new (in the model's generate units) for samples of up to `seconds` of audio; rollout.py
        caps each batch with it, since longer samples fail the duration gate anyway. None = no cap."""
        return None

    def decode(self, samples):
        """[Sample] -> [mono float32 numpy wav or None (undecodable)], at decoded_sample_rate."""
        raise NotImplementedError

    # ---- training --------------------------------------------------------------------------
    def trainable_parameters(self):
        raise NotImplementedError

    def collate(self, examples):
        """examples: [{"prompt", "chosen", "chosen_ended", "rejected", "rejected_ended"}] ->
        a batch dict for sequence_logps (first half chosen, second half rejected) that also has
        "n_frames": FloatTensor [2B] (frames per completion)."""
        raise NotImplementedError

    def sequence_logps(self, batch, return_entropy=False):
        """log pi_theta(completion | prompt) per sequence, FloatTensor [2B] with grad, temperature 1,
        over the same support the sampler draws from. return_entropy=True also returns a 1-D
        tensor of mean per-position entropies (per codebook / slot where it applies)."""
        raise NotImplementedError

    def reference_logps(self, batch):
        """log pi_ref(...) per sequence, no grad: the frozen init_dir model (a second model, or the
        policy with its LoRA adapters disabled). Same code path and precision as sequence_logps."""
        raise NotImplementedError

    def incremental_logp(self, prompt, completion, ended):
        """gates.py D1: the same log-prob recomputed step by step the way the sampler runs (KV
        cache, one token/frame at a time), with no grad. A float."""
        raise NotImplementedError

    def save_checkpoint(self, out_dir):
        """A dir the benchmark generator (cfg["eval_generator"]) loads as --model, plus whatever
        the next round's for_training load needs. Write to a temp dir and rename (atomic)."""
        raise NotImplementedError

    def train(self):
        """Training mode (dropout must already be 0 or disabled: gates.py D3)."""
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def dropout_probs(self):
        """{module name: p} for every dropout in the trainable model (gates.py D3 wants all 0)."""
        raise NotImplementedError
