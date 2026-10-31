"""TTS API configs (apis/*.yaml): loading, validation and the HTTP calls. Standard library + PyYAML.

A config describes one request per sentence. String values may use
  {text} {ref_text} {ref_audio_base64} {voice_id}      filled per sentence
  ${ENV_VAR}  ${ENV_VAR:-default}                       read from the environment at request time
Keys come only from environment variables: they are never written to files or printed.

  name: my-provider
  url: https://api.example.com/v1/tts
  method: POST                      # default POST
  headers: {Authorization: "Bearer ${MY_TTS_API_KEY}"}
  query: {output_format: pcm_24000} # optional URL query parameters
  body: {text: "{text}", reference_audio: "{ref_audio_base64}", format: wav}   # sent as JSON
  response:
    audio: binary                   # or a JSON path to base64 audio, e.g. data.audio_base64
    format: wav                     # wav | mp3 | flac | ogg | pcm_s16le
    sample_rate: 24000              # pcm_s16le only
  voice_cloning: per_request        # per_request | create_voice | none
  create_voice:                     # create_voice only: called once per reference clip
    url: https://api.example.com/v1/voices
    headers: {...}
    form: {name: "turing-tester-{voice_key}"}   # multipart fields; the clip is sent as file field `file_field`
    file_field: files
    voice_id: voice_id              # JSON path of the new voice's id in the response
  delete_voice: {method: DELETE, url: "https://api.example.com/v1/voices/{voice_id}", headers: {...}}
  requests_per_minute: 60
  timeout_seconds: 120
"""
import base64
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

CLONING = ("per_request", "create_voice", "none")
FORMATS = ("wav", "mp3", "flac", "ogg", "pcm_s16le")
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_SLOT = re.compile(r"\{(text|ref_text|ref_audio_base64|voice_id|voice_key)\}")


class ApiError(Exception):
    pass


def _fail(path, msg):
    try:
        from turing_tester.util import Fail
        raise Fail(f"{Path(path).name}: {msg}")
    except ImportError:
        raise SystemExit(f"{Path(path).name}: {msg}")


def load(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            _fail(path, f"not valid YAML: {e}")
    if not isinstance(cfg, dict):
        _fail(path, "must be a YAML mapping")
    for key in ("url", "body", "response"):
        if key not in cfg:
            _fail(path, f"`{key}` is missing (see apis/template.yaml)")
    resp = cfg["response"]
    if not isinstance(resp, dict) or "audio" not in resp:
        _fail(path, "`response.audio` must be `binary` or a JSON path to base64 audio")
    fmt = resp.get("format", "wav")
    if fmt not in FORMATS:
        _fail(path, f"response.format must be one of {', '.join(FORMATS)}")
    if fmt == "pcm_s16le" and not isinstance(resp.get("sample_rate"), int):
        _fail(path, "response.sample_rate is required for pcm_s16le")
    cloning = cfg.get("voice_cloning", "per_request")
    if cloning not in CLONING:
        _fail(path, f"voice_cloning must be one of {', '.join(CLONING)}")
    if cloning == "create_voice" and not isinstance(cfg.get("create_voice"), dict):
        _fail(path, "voice_cloning: create_voice needs a `create_voice` section")
    cfg.setdefault("name", Path(path).stem)
    cfg.setdefault("method", "POST")
    cfg["voice_cloning"] = cloning
    return cfg


def missing_env(cfg):
    """Environment variables the config uses without a default that are not set."""
    found = set()

    def walk(v):
        if isinstance(v, str):
            for m in _ENV.finditer(v):
                if m.group(2) is None and not os.environ.get(m.group(1)):
                    found.add(m.group(1))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(cfg)
    return sorted(found)


def expand(value, slots):
    """Fill {slot} and ${ENV} placeholders in every string of a nested value."""
    if isinstance(value, str):
        def env(m):
            v = os.environ.get(m.group(1))
            if v is None or v == "":
                if m.group(2) is None:
                    raise ApiError(f"environment variable {m.group(1)} is not set")
                return m.group(2)
            return v
        out = _ENV.sub(env, value)
        full = _SLOT.fullmatch(out)
        if full:       # a lone slot keeps the slot value's type
            return slots.get(full.group(1), "")
        return _SLOT.sub(lambda m: str(slots.get(m.group(1), "")), out)
    if isinstance(value, dict):
        return {k: expand(v, slots) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v, slots) for v in value]
    return value


def json_path(obj, path):
    for part in str(path).split("."):
        if isinstance(obj, list):
            obj = obj[int(part)]
        elif isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            raise ApiError(f"response has no `{path}`")
    return obj


def _multipart(fields, file_field, file_path):
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    data = Path(file_path).read_bytes()
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
                 f"filename=\"{Path(file_path).name}\"\r\nContent-Type: audio/wav\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class Client:
    def __init__(self, cfg, log=print):
        self.cfg = cfg
        self.log = log
        rpm = float(cfg.get("requests_per_minute") or 0)
        self.min_gap = 60.0 / rpm if rpm > 0 else 0.0
        self.last = 0.0
        self.timeout = float(cfg.get("timeout_seconds", 120))
        self.retries = int(cfg.get("retries", 5))

    def _http(self, method, url, headers, body=None, content_type=None):
        """One request with rate limiting and retries on 429 / 5xx / network errors. -> (bytes, content-type)"""
        for attempt in range(self.retries + 1):
            wait = self.last + self.min_gap - time.time()
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            h = dict(headers or {})
            if content_type:
                h["Content-Type"] = content_type
            req = urllib.request.Request(url, data=body, headers=h, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return r.read(), r.headers.get("Content-Type", "")
            except urllib.error.HTTPError as e:
                detail = e.read()[:300].decode("utf-8", "replace")
                if e.code in (408, 409, 425, 429) or e.code >= 500:
                    delay = min(60, 2 ** attempt * 2)
                    self.log(f"  [api] HTTP {e.code}, retrying in {delay}s ({detail.strip()[:120]})")
                    time.sleep(delay)
                    continue
                raise ApiError(f"HTTP {e.code} from {urllib.parse.urlsplit(url).netloc}: {detail}")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                delay = min(60, 2 ** attempt * 2)
                self.log(f"  [api] {type(e).__name__}: {e}; retrying in {delay}s")
                time.sleep(delay)
        raise ApiError(f"gave up after {self.retries + 1} attempts")

    def _url(self, section, slots):
        url = expand(section["url"], slots)
        q = expand(section.get("query") or {}, slots)
        if q:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(q)
        return url

    def create_voice(self, ref_wav, voice_key):
        cv = self.cfg["create_voice"]
        slots = {"voice_key": voice_key}
        fields = {k: str(v) for k, v in expand(cv.get("form") or {"name": "turing-tester-{voice_key}"}, slots).items()}
        body, ctype = _multipart(fields, cv.get("file_field", "files"), ref_wav)
        data, _ = self._http(cv.get("method", "POST"), self._url(cv, slots), expand(cv.get("headers") or {}, slots),
                             body, ctype)
        return str(json_path(json.loads(data.decode("utf-8")), cv.get("voice_id", "voice_id")))

    def delete_voice(self, voice_id):
        dv = self.cfg.get("delete_voice")
        if not dv:
            return
        slots = {"voice_id": voice_id}
        try:
            self._http(dv.get("method", "DELETE"), self._url(dv, slots), expand(dv.get("headers") or {}, slots))
        except ApiError as e:
            self.log(f"  [api] could not delete voice {voice_id}: {e}")

    def synthesize(self, text, ref_wav=None, ref_text=None, voice_id=None):
        """-> (audio bytes, format)"""
        slots = {"text": text, "ref_text": ref_text or "", "voice_id": voice_id or ""}
        if ref_wav and self.cfg["voice_cloning"] == "per_request":
            slots["ref_audio_base64"] = base64.b64encode(Path(ref_wav).read_bytes()).decode("ascii")
        body = json.dumps(expand(self.cfg["body"], slots)).encode("utf-8")
        data, ctype = self._http(self.cfg["method"], self._url(self.cfg, slots),
                                 expand(self.cfg.get("headers") or {}, slots), body, "application/json")
        resp = self.cfg["response"]
        if resp["audio"] != "binary":
            if "json" not in ctype and not data.lstrip().startswith(b"{"):
                raise ApiError(f"expected a JSON response, got {ctype or 'unknown content'}")
            data = base64.b64decode(json_path(json.loads(data.decode("utf-8")), resp["audio"]))
        elif data.lstrip()[:1] == b"{" and "json" in ctype:
            raise ApiError(f"expected audio, got JSON: {data[:200].decode('utf-8', 'replace')}")
        return data, resp.get("format", "wav")
