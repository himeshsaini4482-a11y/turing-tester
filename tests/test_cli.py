"""CPU tests for the voicegym CLI: model detection, API configs and calls (against a local fake
server), the custom and API generators, voice preparation, reports and resumable steps.

Needs numpy, soundfile, librosa and pyyaml (no GPU, no model weights):
  python -m unittest discover -s tests -v
"""
import base64
import csv
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from voicegym_cli import apiconfig, models, report, util  # noqa: E402
from voicegym_cli.util import Fail  # noqa: E402

PY = sys.executable


def sine_wav(path, seconds=2.0, sr=16000, freq=220.0):
    n = int(seconds * sr)
    frames = b"".join(int(8000 * math.sin(2 * math.pi * freq * i / sr)).to_bytes(2, "little", signed=True)
                      for i in range(n))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(frames)
    return path


def wav_bytes(seconds=1.0, sr=16000):
    buf = io.BytesIO()
    n = int(seconds * sr)
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(int(5000 * math.sin(i / 10)).to_bytes(2, "little", signed=True) for i in range(n)))
    return buf.getvalue()


def write_manifest(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run_script(script, *args, env=None):
    e = dict(os.environ, PYTHONPATH="", PYTHONIOENCODING="utf-8")
    e.update(env or {})
    return subprocess.run([PY, str(REPO / script), *map(str, args)], cwd=REPO, env=e, capture_output=True, text=True,
                          encoding="utf-8")


class Detection(unittest.TestCase):
    def folder(self, config=None, files=None):
        d = Path(tempfile.mkdtemp())
        if config is not None:
            (d / "config.json").write_text(json.dumps(config))
        for name, text in (files or {}).items():
            (d / name).write_text(text)
        return d

    def test_builtin_families(self):
        self.assertEqual(models.detect(str(self.folder({"model_type": "qwen3_tts", "tts_model_type": "base"}))).kind,
                         "qwen3_tts")
        m = models.detect(str(self.folder({"model_type": "qwen3_tts", "tts_model_type": "custom_voice",
                                           "talker_config": {"spk_id": {"me": 3000}}})))
        self.assertEqual(m.custom_voice_speaker, "me")
        self.assertIn("--speaker", m.generate_cmd("m.csv", "out")[1])
        self.assertEqual(models.detect(str(self.folder({"model_type": "moss_tts_nano"}))).kind, "moss_tts_nano")
        self.assertEqual(models.detect(str(self.folder({"model_type": "llama", "vocab_size": 156939}))).kind, "orpheus")

    def test_capabilities(self):
        moss = models.detect(str(self.folder({"model_type": "moss_tts_nano"})))
        self.assertTrue(moss.can_sft and moss.can_dpo)
        qwen = models.detect(str(self.folder({"model_type": "qwen3_tts"})))
        self.assertTrue(qwen.can_sft)
        self.assertFalse(qwen.can_dpo)
        custom = models.detect(str(self.folder(None, {"synthesize.py": "def synthesize(t, a, b): pass"})))
        self.assertEqual(custom.kind, "custom")
        self.assertFalse(custom.can_sft or custom.can_dpo)
        trainable = models.detect(str(self.folder(None, {"synthesize.py": "", "train.py": ""})))
        self.assertTrue(trainable.can_sft)

    def test_unknown_and_missing(self):
        with self.assertRaises(Fail):
            models.detect(str(self.folder({"model_type": "bert"})))
        with self.assertRaises(Fail) as cm:
            models.detect("models/does-not-exist")
        self.assertIn("not found", str(cm.exception.code))

    def test_token_lm_yaml(self):
        good = ("type: token-lm\ncodec: snac_24khz\naudio_token_offset: 128266\nstop_token: 128258\n"
                "prompt:\n  - [128259]\n  - ref_text\n  - [128009]\n  - ref_audio\n  - [128259]\n  - text\n  - [128009]\n")
        m = models.detect(str(self.folder({"model_type": "llama", "vocab_size": 5}, {"voicegym.yaml": good})))
        self.assertEqual(m.kind, "token_lm")
        self.assertTrue(m.can_dpo)
        for bad in (good.replace("codec: snac_24khz", "codec: nope.py"), good.replace("  - text\n", ""),
                    good.replace("stop_token: 128258", "stop_token: x")):
            with self.assertRaises(Fail):
                models.detect(str(self.folder({"model_type": "llama"}, {"voicegym.yaml": bad})))
        forced = models.detect(str(self.folder({"model_type": "llama"}, {"voicegym.yaml": "family: orpheus\n"})))
        self.assertEqual(forced.kind, "orpheus")


class ApiConfigs(unittest.TestCase):
    def test_bundled_configs_load(self):
        for name in ("template", "elevenlabs", "openai-compatible"):
            cfg = apiconfig.load(REPO / "apis" / f"{name}.yaml")
            self.assertIn(cfg["voice_cloning"], apiconfig.CLONING)
        self.assertEqual(apiconfig.load(REPO / "apis" / "elevenlabs.yaml")["voice_cloning"], "create_voice")
        self.assertEqual(apiconfig.load(REPO / "apis" / "openai-compatible.yaml")["voice_cloning"], "none")

    def test_env_and_slots(self):
        os.environ.pop("VG_TEST_KEY", None)
        cfg = {"headers": {"Authorization": "Bearer ${VG_TEST_KEY}"}, "url": "${VG_TEST_URL:-http://x}/a"}
        self.assertEqual(apiconfig.missing_env(cfg), ["VG_TEST_KEY"])
        with self.assertRaises(apiconfig.ApiError):
            apiconfig.expand(cfg["headers"], {})
        os.environ["VG_TEST_KEY"] = "secret"
        self.assertEqual(apiconfig.expand(cfg, {})["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(apiconfig.expand(cfg, {})["url"], "http://x/a")
        self.assertEqual(apiconfig.expand({"t": "{text}", "n": "say {text}!"}, {"text": "hi"}), {"t": "hi", "n": "say hi!"})
        self.assertEqual(apiconfig.json_path({"a": [{"b": 3}]}, "a.0.b"), 3)

    def test_bad_config(self):
        d = Path(tempfile.mkdtemp())
        (d / "x.yaml").write_text("url: http://x\nbody: {}\nresponse: {audio: binary, format: aiff}\n")
        with self.assertRaises(Fail):
            apiconfig.load(d / "x.yaml")


class FakeTTS(BaseHTTPRequestHandler):
    calls = []
    fail_first = {"n": 0}

    def log_message(self, *a):
        pass

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def do_POST(self):
        body = self._body()
        FakeTTS.calls.append((self.path, dict(self.headers), body))
        if FakeTTS.fail_first["n"] > 0:
            FakeTTS.fail_first["n"] -= 1
            self.send_response(503)
            self.end_headers()
            return
        if self.path.startswith("/voices/add"):
            out = json.dumps({"voice_id": f"v{len(FakeTTS.calls)}"}).encode()
            ctype = "application/json"
        elif self.path.startswith("/json"):
            out = json.dumps({"data": {"audio": base64.b64encode(wav_bytes()).decode()}}).encode()
            ctype = "application/json"
        else:
            if self.headers.get("Authorization") != "Bearer k123" and self.headers.get("xi-api-key") != "k123":
                self.send_response(401)
                self.end_headers()
                return
            out, ctype = wav_bytes(), "audio/wav"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_DELETE(self):
        FakeTTS.calls.append((self.path, dict(self.headers), b""))
        self.send_response(200)
        self.end_headers()


class ApiGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeTTS)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        FakeTTS.calls.clear()
        self.d = Path(tempfile.mkdtemp())
        rows = []
        for i in range(3):
            spk = f"s{i % 2}"
            rows.append({"speaker_id": spk, "sentence_id": f"{spk}_{i}", "text": f"Sentence number {i}.",
                         "human_wav": str(sine_wav(self.d / f"h{i}.wav")), "prompt_id": f"{spk}_p",
                         "prompt_wav": str(sine_wav(self.d / f"p{spk}.wav", 1.0)), "prompt_text": "Prompt."})
        write_manifest(self.d / "m.csv", rows)

    def config(self, text):
        p = self.d / "api.yaml"
        p.write_text(text.replace("PORT", str(self.port)))
        return p

    def test_per_request_with_retry(self):
        FakeTTS.fail_first["n"] = 1
        cfg = self.config('name: fake\nurl: "http://127.0.0.1:PORT/tts"\nheaders: {Authorization: "Bearer ${VG_FAKE_KEY}"}\n'
                          'body: {text: "{text}", ref: "{ref_audio_base64}", rt: "{ref_text}"}\n'
                          "response: {audio: binary, format: wav}\nvoice_cloning: per_request\nretries: 2\n")
        r = run_script("benchmark/generate_api.py", "--manifest", self.d / "m.csv", "--run-dir", self.d / "run",
                       "--config", cfg, env={"VG_FAKE_KEY": "k123"})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("k123", r.stdout + r.stderr)
        self.assertEqual(len(list((self.d / "run" / "tts").glob("*.wav"))), 3)
        posted = [json.loads(b) for p, _, b in FakeTTS.calls if p == "/tts" and b]
        self.assertTrue(all(base64.b64decode(x["ref"])[:4] == b"RIFF" for x in posted))
        self.assertTrue((self.d / "run" / "manifest.csv").exists())

    def test_missing_key(self):
        cfg = self.config('url: "http://127.0.0.1:PORT/tts"\nheaders: {Authorization: "Bearer ${VG_NOT_SET_KEY}"}\n'
                          'body: {text: "{text}"}\nresponse: {audio: binary}\n')
        r = run_script("benchmark/generate_api.py", "--manifest", self.d / "m.csv", "--run-dir", self.d / "run",
                       "--config", cfg, env={"VG_NOT_SET_KEY": ""})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("VG_NOT_SET_KEY", r.stderr)

    def test_create_voice_json_audio_and_delete(self):
        cfg = self.config('url: "http://127.0.0.1:PORT/json/{voice_id}"\nbody: {text: "{text}"}\n'
                          "response: {audio: data.audio, format: wav}\nvoice_cloning: create_voice\n"
                          'create_voice: {url: "http://127.0.0.1:PORT/voices/add", form: {name: "vg-{voice_key}"}, '
                          "file_field: files, voice_id: voice_id}\n"
                          'delete_voice: {url: "http://127.0.0.1:PORT/voices/{voice_id}"}\n')
        r = run_script("benchmark/generate_api.py", "--manifest", self.d / "m.csv", "--run-dir", self.d / "run",
                       "--config", cfg)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        paths = [p for p, _, _ in FakeTTS.calls]
        self.assertEqual(sum(p == "/voices/add" for p in paths), 2)       # one voice per prompt clip
        self.assertEqual(sum(p.startswith("/voices/v") for p in paths), 2)  # both deleted
        self.assertTrue(any(p.startswith("/json/v") for p in paths))
        self.assertFalse((self.d / "run" / "api_voices.json").exists())


class CustomGenerator(unittest.TestCase):
    def test_synthesize_hook(self):
        d = Path(tempfile.mkdtemp())
        model = d / "model"
        model.mkdir()
        (model / "helper.py").write_text("SR = 8000\n")
        (model / "synthesize.py").write_text(
            "import numpy as np\nfrom helper import SR\nloaded = {}\n"
            "def load(device):\n    loaded['device'] = device\n"
            "def synthesize(text, ref_audio, ref_text):\n    assert ref_audio and loaded\n"
            "    return np.zeros(SR // 2, dtype='float32') + 0.01 * len(text), SR\n")
        sine_wav(d / "p.wav")
        write_manifest(d / "m.csv", [{"speaker_id": "a", "sentence_id": "a_1", "text": "Hello there.", "human_wav": "",
                                      "prompt_wav": str(d / "p.wav"), "prompt_text": "x"}])
        r = run_script("benchmark/generate_custom.py", "--manifest", d / "m.csv", "--run-dir", d / "run",
                       "--model", model, "--device", "cpu")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        import soundfile as sf
        info = sf.info(str(d / "run" / "tts" / "a_1.wav"))
        self.assertEqual(info.samplerate, 8000)
        self.assertAlmostEqual(info.duration, 0.5, places=2)


class VoicePreparation(unittest.TestCase):
    def test_split_and_records(self):
        d = Path(tempfile.mkdtemp())
        voice = d / "my-voice"
        voice.mkdir()
        lines = ["file,text"]
        for i in range(60):
            sine_wav(voice / f"clip_{i:03d}.wav", seconds=3 + (i % 8), freq=150 + i)
            lines.append(f"clip_{i:03d}.wav,This is test sentence number {i}.")
        sine_wav(voice / "too_short.wav", seconds=0.5)
        (voice / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        out = d / "prepared"
        r = run_script("benchmark/prepare_voice.py", "--voice", voice, "--out", out, "--device", "cpu")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        s = json.loads((out / "summary.json").read_text())
        self.assertEqual((s["n_clips"], s["n_test"], s["n_dev"], s["n_train"]), (60, 20, 10, 30))
        self.assertEqual(s["skipped"]["too_short"], 1)
        self.assertEqual(s["transcripts"]["metadata.csv"], 60)
        with open(out / "test.csv", encoding="utf-8") as f:
            test = list(csv.DictReader(f))
        with open(out / "dev.csv", encoding="utf-8") as f:
            dev = list(csv.DictReader(f))
        with open(out / "clips.jsonl", encoding="utf-8") as f:
            prompt = next(c for c in map(json.loads, f) if c["prompt"])
        self.assertTrue(all(t["prompt_wav"] == prompt["wav"] and t["human_wav"] != prompt["wav"] for t in test + dev))
        with open(out / "train.jsonl", encoding="utf-8") as f:
            train = [json.loads(x) for x in f]
        self.assertEqual(len(train), 30)
        test_wavs, dev_wavs, train_wavs = ({t["human_wav"] for t in test}, {t["human_wav"] for t in dev},
                                           {x["audio"] for x in train})
        self.assertFalse(test_wavs & dev_wavs or test_wavs & train_wavs or dev_wavs & train_wavs)

    def test_too_few_clips(self):
        d = Path(tempfile.mkdtemp())
        (d / "v").mkdir()
        for i in range(10):
            sine_wav(d / "v" / f"c{i}.wav", 3)
        (d / "v" / "metadata.csv").write_text("\n".join(f"c{i}.wav,Hello number {i}." for i in range(10)))
        r = run_script("benchmark/prepare_voice.py", "--voice", d / "v", "--out", d / "o", "--device", "cpu")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("at least 50", r.stderr)


class Reports(unittest.TestCase):
    def make_run(self, d, tts_shift=0.0, n=40):
        d.mkdir(parents=True)
        rows, j, b, m = [], [], [], []
        for i in range(n):
            sid, spk = f"s{i}", f"p{i % 2}"
            rows.append({"speaker_id": spk, "sentence_id": sid, "text": "x"})
            for order in ("tts_first", "human_first"):
                j.append({"speaker_id": spk, "sentence_id": sid, "order": order, "vote": 0,
                          "tts_score": 7.0 + tts_shift, "human_score": 8.0, "analysis": ""})
            b.append({"speaker_id": spk, "sentence_id": sid, "tts_reward": 2.0 + tts_shift + (i % 3) * 0.1,
                      "human_reward": 4.0, "p_tts_better": 0.2})
            m.append({"sentence_id": sid, "speaker_id": spk, "tts_wer": 0.05, "human_wer": 0.02,
                      "speaker_sim": 0.9, "tts_utmos": 3.5, "human_utmos": 4.0})
        write_manifest(d / "manifest.csv", rows)
        for name, data in (("judgments.jsonl", j), ("btrm.jsonl", b), ("metrics.jsonl", m)):
            (d / name).write_text("\n".join(json.dumps(x) for x in data) + "\n")
        return d

    def test_benchmark_report(self):
        d = self.make_run(Path(tempfile.mkdtemp()) / "run")
        s = report.benchmark_report(d, {"name": "t", "judges": ["grm", "btrm"], "model_label": "X", "model": "m"})
        self.assertAlmostEqual(s["grm_closeness"][0], 87.5)
        self.assertAlmostEqual(s["btrm_pct"][0], 100 * (2.0 + 0.1 * (13 + 13 * 2) / 40) / 4.0, places=4)
        self.assertLess(s["btrm_pct"][1], s["btrm_pct"][0])
        text = (d / "report.md").read_text(encoding="utf-8")
        self.assertIn("GRM closeness", text)
        self.assertIn(report.DISCLAIMER, text)
        self.assertTrue((d / "report.json").exists())

    def test_undefined_btrm_and_comparison(self):
        root = Path(tempfile.mkdtemp())
        a = self.make_run(root / "a")
        b = self.make_run(root / "b", tts_shift=1.0)
        lines = report.comparison_lines(a, b, "base", "new")
        self.assertIn("clear gain", "\n".join(lines))
        rows = [{"r_tts": 1.0, "r_human": -1.0}]
        self.assertIsNone(report.btrm_pct(rows))


class Naming(unittest.TestCase):
    def test_default_names_from_run_folders(self):
        root = Path(tempfile.mkdtemp())
        for sub in ("sft/me/final", "dpo/me/ckpt/r02"):
            d = root / sub
            d.mkdir(parents=True)
            (d / "config.json").write_text(json.dumps({"model_type": "moss_tts_nano"}))
        self.assertEqual(models.detect(str(root / "sft/me/final")).default_name, "me_final")
        self.assertEqual(models.detect(str(root / "dpo/me/ckpt/r02")).default_name, "me_r02")

    def test_user_texts_drop_benchmark_sentences(self):
        from voicegym_cli import sft
        bench = next(csv.DictReader(open(REPO / "benchmark/manifests/ears_regular_26x19.csv", encoding="utf-8")))["text"]
        p = Path(tempfile.mkdtemp()) / "t.txt"
        p.write_text("A perfectly ordinary new sentence about the weather today." + chr(10) + bench + chr(10), encoding="utf-8")
        self.assertEqual(sft.user_texts(p), ["A perfectly ordinary new sentence about the weather today."])


class StepsAndCli(unittest.TestCase):
    def test_steps_resume_and_key(self):
        d = Path(tempfile.mkdtemp())
        s, calls = util.Steps(d), []
        s("a", lambda: calls.append(1), key="k1")
        s("a", lambda: calls.append(2), key="k1")
        s("a", lambda: calls.append(3), key="k2")
        self.assertEqual(calls, [1, 3])

    def test_cli_errors(self):
        from voicegym_cli import cli
        self.assertEqual(cli.main(["benchmark"]), 2)
        self.assertEqual(cli.main(["benchmark", "--model", "x", "--api", "y"]), 2)
        self.assertEqual(cli.main([]), 0)

    def test_pool(self):
        sys.path.insert(0, str(REPO / "sft"))
        import gen_texts
        rows = [json.loads(x) for x in open(REPO / "voicegym_cli" / "data" / "sentence_pool.jsonl", encoding="utf-8")]
        self.assertGreater(len(rows), 2000)
        self.assertTrue(all(gen_texts.invalid_reason(r["text"]) is None for r in rows))
        self.assertEqual(len({r["id"] for r in rows}), len(rows))


if __name__ == "__main__":
    unittest.main()
