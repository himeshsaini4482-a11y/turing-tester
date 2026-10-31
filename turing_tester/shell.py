"""The interactive interface: ./turing-tester with no command opens it. You type / commands; the
menu under the prompt suggests the closest ones (slash.py ranks them). The commands run the same
code as the regular ones (cli.py), in this process, with a live status line under their output.

Needs prompt_toolkit and rich, which the launcher installs.
"""
import os
import re
import sys
import time
import traceback
import wave
from contextlib import contextmanager
from pathlib import Path

from turing_tester import paths, slash, util
from turing_tester.slash import BY_NAME, COMMANDS

ACCENT = "#5fafaf"
STYLE = {
    "prompt": f"{ACCENT} bold",
    "placeholder": "#808080",
    "bottom-toolbar": "noreverse #808080",
    "completion-menu": "bg:#262626 #d0d0d0",
    "completion-menu.completion": "bg:#262626 #d0d0d0",
    "completion-menu.completion.current": f"bg:{ACCENT} #000000 bold",
    "completion-menu.meta.completion": "bg:#262626 #8a8a8a",
    "completion-menu.meta.completion.current": f"bg:{ACCENT} #000000",
    "q": "bold",
    "sel": f"{ACCENT} bold",
    "hint": "#808080",
}


def available():
    try:
        import prompt_toolkit  # noqa: F401
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


# ---- suggestions under the prompt ---------------------------------------------------------------

def _subparser(name):
    from turing_tester import cli
    sub = next(a for a in cli.build_parser()._actions if a.__class__.__name__ == "_SubParsersAction")
    return sub.choices.get(name)


def _options(cmd_name):
    """[(--option, help, takes_value)] of a regular command, hidden ones left out."""
    p = _subparser(cmd_name)
    if p is None:
        return []
    out = []
    for a in p._actions:
        if not a.option_strings or a.help == "==SUPPRESS==" or "-h" in a.option_strings:
            continue
        out.append((a.option_strings[-1], a.help or "", a.nargs != 0))
    return out


def _rel(p):
    return paths.rel(p) + ("/" if Path(p).is_dir() else "")


def _value_candidates(cmd, option, word):
    if option in ("--model", "--teacher"):
        vals = [_rel(p) for p in slash.model_folders()]
        if option == "--teacher":
            vals += [_rel(p) for p in slash.api_configs()]
    elif option == "--api":
        vals = [_rel(p) for p in slash.api_configs()]
    elif option == "--voice":
        vals = [_rel(p) for p in slash.voice_folders()]
    elif option == "--set":
        vals = ["ears", "libritts-r"] + [_rel(p) for p in slash.voice_folders()]
    elif option in slash.CHOICES:
        vals = slash.CHOICES[option]
    else:
        vals = []
    vals = [v for v in vals if v.lower().startswith(word.lower())]
    return vals or slash.path_candidates(word)


def _make_completer():
    from prompt_toolkit.completion import Completer, Completion

    class SlashCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/"):
                return
            if " " not in text:                    # the command name: closest commands first
                word = text[1:]
                for c in slash.rank_commands(word):
                    yield Completion(c.name + " ", start_position=-len(word), display="/" + c.name,
                                     display_meta=c.about)
                return
            words = text.split(" ")
            cmd = slash.resolve(words[0])
            if cmd is None:
                return
            word, prev = words[-1], (words[-2] if len(words) > 2 else "")
            for value, meta in self._args(cmd, word, prev, words[1:-1]):
                yield Completion(value, start_position=-len(word), display_meta=meta)

        def _args(self, cmd, word, prev, before):
            opts = _options(cmd.name)
            takes_value = {o: v for o, _, v in opts}
            if word.startswith("-"):
                names = [o.lstrip("-") for o, _, _ in opts]
                helps = {o.lstrip("-"): h for o, h, _ in opts}
                return [("--" + n, helps[n]) for n in slash.rank(word.lstrip("-"), names)]
            if prev.startswith("-") and takes_value.get(prev):
                return [(v, "") for v in _value_candidates(cmd, prev, word)]
            if cmd.name == "setup":
                return [(e, "environment") for e in slash.SETUP_ENVS if e.startswith(word) and e not in before]
            if cmd.name == "help":
                return [(c.name, c.about) for c in slash.rank_commands(word)]
            if cmd.name == "report":
                return [(r["id"], r["kind"]) for r in list_runs() if r["id"].lower().startswith(word.lower())]
            if cmd.name in ("benchmark", "sft", "dpo") and not [b for b in before if b]:
                vals = [_rel(p) for p in slash.model_folders()]
                if cmd.name == "benchmark":
                    vals += [_rel(p) for p in slash.api_configs()]
                vals = [v for v in vals if v.lower().startswith(word.lower())]
                if vals:
                    return [(v, "model" if not v.endswith((".yaml", ".yml")) else "API") for v in vals]
            return [(p, "") for p in slash.path_candidates(word)]

    return SlashCompleter()


def _key_bindings():
    from prompt_toolkit.filters import completion_is_selected, has_completions
    from prompt_toolkit.key_binding import KeyBindings

    kb = KeyBindings()

    @kb.add("enter", filter=completion_is_selected)
    def _(event):
        """Enter on a highlighted suggestion: a command runs; an option or path is filled in."""
        b = event.current_buffer
        b.apply_completion(b.complete_state.current_completion)
        if b.text.startswith("/") and b.text.count(" ") <= 1 and b.text.endswith(" "):
            b.validate_and_handle()

    @kb.add("tab")
    def _(event):
        """Tab takes the highlighted (or first) suggestion; with no menu open, opens it."""
        b = event.current_buffer
        st = b.complete_state
        if st and st.completions:
            b.apply_completion(st.current_completion or st.completions[0])
        else:
            b.start_completion(select_first=True)

    @kb.add("escape", eager=True, filter=has_completions)
    def _(event):
        event.current_buffer.cancel_completion()

    @kb.add("c-c")
    def _(event):
        """Ctrl+C clears what you typed; at an empty prompt it asks to leave (twice leaves)."""
        b = event.current_buffer
        if b.text:
            b.reset()
        else:
            event.app.exit(exception=KeyboardInterrupt())

    return kb


# ---- questions for the guided commands ---------------------------------------------------------

class Cancelled(Exception):
    pass


def select(console, title, options, default=0):
    """Arrow-key choice. options: [(value, label, hint)]. Esc or Ctrl+C -> Cancelled."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    idx = [min(default, len(options) - 1)]

    def text():
        out = [("class:q", f"{title}\n")]
        for i, (_, label, hint) in enumerate(options):
            cur = i == idx[0]
            out.append(("class:sel" if cur else "", f" {'>' if cur else ' '} {i + 1}. {label}"))
            if hint:
                out.append(("class:hint", f"   {hint}"))
            out.append(("", "\n"))
        out.append(("class:hint", " Up/Down choose · Enter select · Esc cancel"))
        return out

    kb = KeyBindings()

    @kb.add("up")
    def _(e):
        idx[0] = (idx[0] - 1) % len(options)

    @kb.add("down")
    def _(e):
        idx[0] = (idx[0] + 1) % len(options)

    @kb.add("enter")
    def _(e):
        e.app.exit(result=idx[0])

    for n in range(1, min(len(options), 9) + 1):
        @kb.add(str(n))
        def _(e, n=n):
            e.app.exit(result=n - 1)

    @kb.add("escape", eager=True)
    @kb.add("c-c")
    def _(e):
        e.app.exit(result=None)

    app = Application(layout=Layout(Window(FormattedTextControl(text, focusable=True), always_hide_cursor=True)),
                      key_bindings=kb, style=Style.from_dict(STYLE), erase_when_done=True)
    i = app.run()
    if i is None:
        raise Cancelled
    console.print(f"[bold]{title}[/] [{ACCENT}]{options[i][1]}[/]", highlight=False)
    return options[i][0]


def ask(console, question, default="", paths_only=False):
    """One line of text (Tab completes repo paths). Esc or Ctrl+C -> Cancelled."""
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style

    class Paths(Completer):
        def get_completions(self, document, complete_event):
            word = document.text_before_cursor
            for p in slash.path_candidates(word):
                yield Completion(p, start_position=-len(word))

    kb = KeyBindings()

    @kb.add("escape", eager=True)
    def _(e):
        e.app.exit(result=None)

    try:
        ans = prompt([("class:q", question + " ")], default=default, key_bindings=kb,
                     completer=Paths() if paths_only else None, complete_while_typing=paths_only,
                     style=Style.from_dict(STYLE))
    except (KeyboardInterrupt, EOFError):
        ans = None
    if ans is None or not ans.strip():
        raise Cancelled
    return ans.strip()


def _model_label(path):
    """(label, Model or None) for a folder, without raising."""
    from turing_tester import models
    try:
        m = models.detect(str(path))
        return m.label, m
    except SystemExit as e:              # util.Fail
        return "not recognised: " + str(e.code).split(":", 1)[-1].strip().splitlines()[0][:60], None


def _pick_model(console, title, need=None, with_apis=False):
    """Choose a model folder (or an API config); `need` = "sft" | "dpo" hides models that cannot do it."""
    from turing_tester import models
    opts = []
    for p in slash.model_folders():
        label, m = _model_label(p)
        if need and m is not None and not getattr(m, f"can_{need}"):
            continue
        opts.append((_rel(p), _rel(p), label))
    if with_apis:
        for a in slash.api_configs():
            try:
                name = models.detect_api(str(a)).api.get("name") or a.stem
            except SystemExit:
                name = a.stem
            opts.append((_rel(a), _rel(a), f"TTS API ({name})"))
    if not opts:
        console.print("[dim]models/ is empty: copy a model folder into it, or type a Hugging Face id.[/]")
    opts += [("__hf", "A Hugging Face id…", "e.g. OpenMOSS-Team/MOSS-TTS-Nano-100M"),
             ("__path", "Another folder…", "type its path")]
    v = select(console, title, opts)
    if v == "__hf":
        return ask(console, "Hugging Face id:")
    if v == "__path":
        return ask(console, "Folder:", paths_only=True)
    return v


def _confirm(console, cmd, args):
    line = "/" + " ".join([cmd] + [a if " " not in a else f'"{a}"' for a in args])
    console.print(f"[dim]Same as typing:[/] [bold]{line}[/]", highlight=False)
    if select(console, "Start?", [(True, "Start", ""), (False, "Cancel", "")]) is not True:
        raise Cancelled
    return args


def guided(console, cmd):
    """Ask what a command needs; -> the arguments to run it with."""
    if cmd == "benchmark":
        target = _pick_model(console, "What do you want to benchmark?", with_apis=True)
        sets = [("ears", "EARS", "494 sentences, 26 speakers, studio recordings (the default)"),
                ("libritts-r", "LibriTTS-R", "500 sentences, 10 speakers, audiobook recordings")]
        sets += [(_rel(v), f"Your voice: {_rel(v)}", "its held-out test clips") for v in slash.voice_folders()]
        bset = select(console, "Which sentences?", sets)
        size = select(console, "How many?", [("10", "Quick check: the first 10", "minutes; too few to trust"),
                                              ("0", "The full set", "hours; the real benchmark")])
        args = [target] + (["--set", bset] if bset != "ears" else []) + (["--limit", size] if size != "0" else [])
        return _confirm(console, cmd, args)
    if cmd == "sft":
        student = _pick_model(console, "Which model do you want to train?", need="sft")
        srcs = [("ears", "The EARS human recordings", "the default; studio speech from many speakers")]
        srcs += [(f"voice:{_rel(v)}", f"Your voice: {_rel(v)}", "learn to speak like your recordings")
                 for v in slash.voice_folders()]
        srcs += [("teacher", "A bigger model as teacher", "it speaks new sentences for your model to copy")]
        src = select(console, "What should it learn from?", srcs)
        args = [student]
        if src.startswith("voice:"):
            args += ["--voice", src[6:]]
        elif src == "teacher":
            args += ["--teacher", _pick_model(console, "Which teacher?", with_apis=True)]
        return _confirm(console, cmd, args)
    if cmd == "dpo":
        model = _pick_model(console, "Which model do you want to improve?", need="dpo")
        voices = [("", "The EARS speakers", "the default; improves it across many voices")]
        voices += [(_rel(v), f"Your voice: {_rel(v)}", "aim the improvement at the voice you will deploy")
                   for v in slash.voice_folders()]
        voice = select(console, "Which voices should it practise with?", voices)
        return _confirm(console, cmd, [model] + (["--voice", voice] if voice else []))
    if cmd == "setup":
        v = select(console, "Which environments?", [
            ("tts judge", "Generation and the judges", "what /benchmark needs (the default)"),
            ("tts judge moss lm qwen3-train", "Everything", "also SFT and DPO of every model family"),
            *[(e, e, "") for e in slash.SETUP_ENVS]])
        return v.split()
    return []


# ---- listings ------------------------------------------------------------------------------------

def _rows_scores(report_json):
    j = util.read_json(report_json, {})

    def first(v):
        return v[0] if isinstance(v, list) and v else v
    return {"grm": first(j.get("grm_closeness")), "btrm": first(j.get("btrm_pct")), "wer": j.get("wer")}


def _old_scores(d):
    """Runs made by the experiment scripts (scripts/) keep separate summary files."""
    s = util.read_json(d / "summary.json", {}) or {}
    b = util.read_json(d / "btrm_summary.json", {}) or {}
    m = (util.read_json(d / "metrics_summary.json", {}) or {}).get("overall", {})
    btrm = None
    if b.get("mean_human_reward"):
        btrm = 100 * b["mean_tts_reward"] / b["mean_human_reward"] if b["mean_human_reward"] > 0 else None
    return {"grm": s.get("overall_closeness"), "btrm": btrm, "wer": m.get("tts_wer")}


def list_runs():
    """Every run under runs/, newest first: {"id", "kind", "dir", "grm", "btrm", "wer", "state", "time"}."""
    out = []
    if not paths.RUNS.is_dir():
        return out
    for d in paths.RUNS.iterdir():
        if not d.is_dir() or d.name in ("sft", "dpo"):
            continue
        if (d / "report.json").exists():
            out.append({"id": d.name, "kind": "benchmark", "dir": d, **_rows_scores(d / "report.json"),
                        "state": "done", "time": (d / "report.json").stat().st_mtime})
        elif (d / "turing-tester.json").exists():
            out.append({"id": d.name, "kind": "benchmark", "dir": d, "grm": None, "btrm": None, "wer": None,
                        "state": "stopped", "time": d.stat().st_mtime})
        elif any((d / f).exists() for f in ("summary.json", "btrm_summary.json")):   # experiment scripts
            out.append({"id": d.name, "kind": "benchmark*", "dir": d, **_old_scores(d),
                        "state": "done", "time": d.stat().st_mtime})
    for kind in ("sft", "dpo"):
        root = paths.RUNS / kind
        if not root.is_dir():
            continue
        for d in root.iterdir():
            if not (d / "turing-tester.json").exists():
                continue
            scores = _rows_scores(d / "bench_final" / "report.json") if (d / "bench_final" / "report.json").exists() \
                else {"grm": None, "btrm": None, "wer": None}
            state = "final ready" if (d / "final").is_dir() else ("no gain" if (d / "report.md").exists()
                                                                 else "stopped")
            out.append({"id": f"{kind}/{d.name}", "kind": kind, "dir": d, **scores, "state": state,
                        "time": d.stat().st_mtime})
    return sorted(out, key=lambda r: -r["time"])


def show_runs(console):
    from rich.table import Table
    runs = list_runs()
    if not runs:
        console.print("No runs yet. Start one with [bold]/benchmark[/]. "
                      "[dim](Our own experiment results are in RESULTS.md.)[/]")
        return
    t = Table(box=None, header_style="bold", pad_edge=False)
    t.add_column("Run", no_wrap=True, overflow="ellipsis", min_width=12)
    for col in ("Kind", "GRM", "BTRM %", "WER", "State", "When"):
        t.add_column(col, no_wrap=True, justify="right" if col in ("GRM", "BTRM %", "WER") else "left")
    for r in runs:
        t.add_row(r["id"], r["kind"], _num(r["grm"]), _num(r["btrm"]),
                  "—" if r["wer"] is None else f"{100 * r['wer']:.1f}%", r["state"],
                  time.strftime("%m-%d %H:%M", time.localtime(r["time"])) if r["time"] else "—")
    console.print(t)
    if any(r["kind"] == "benchmark*" for r in runs):
        console.print("[dim]* made by the experiment scripts (scripts/)[/]")
    console.print("[dim]GRM = GRM closeness. Real human = 100 on both judges; automatic judges, not human "
                  "listeners. Details: /report <run>[/]")


def _num(v):
    return "—" if v is None else f"{v:.1f}"


def show_report(console, run_id):
    from rich.markdown import Markdown
    from rich.table import Table
    run_id = run_id.replace("\\", "/").strip("/")
    if run_id.startswith("runs/"):
        run_id = run_id[5:]
    match = next((r for r in list_runs() if r["id"] == run_id), None)
    if match is None:
        ids = [r["id"] for r in list_runs()]
        near = slash.rank(run_id, ids)[:3]
        console.print(f"No run called [bold]{run_id}[/]." + (f" Did you mean: {', '.join(near)}?" if near else
                                                              " See /runs."))
        return
    report = match["dir"] / "report.md"
    if report.exists():
        console.print(Markdown(report.read_text(encoding="utf-8")))
        console.print(f"[dim]{paths.rel(report)}[/]")
        return
    t = Table(box=None, header_style="bold")
    t.add_column("Score")
    t.add_column(run_id, justify="right")
    t.add_column("Real human", justify="right")
    t.add_row("GRM closeness", _num(match["grm"]), "100")
    t.add_row("BTRM %", _num(match["btrm"]), "100")
    t.add_row("WER", "—" if match["wer"] is None else f"{100 * match['wer']:.1f}%", "")
    console.print(t)
    console.print(f"[dim]{match['state']}; files in {paths.rel(match['dir'])}/[/]")


def show_models(console):
    from rich.table import Table
    from turing_tester import apiconfig, models
    folders = slash.model_folders()
    if folders:
        t = Table(box=None, header_style="bold", title="Models", title_justify="left", title_style="bold")
        for col in ("Folder", "Type", "Benchmark", "SFT", "DPO"):
            t.add_column(col)
        for p in folders:
            label, m = _model_label(p)
            yes = lambda ok: "[green]yes[/]" if ok else "[dim]no[/]"  # noqa: E731
            t.add_row(_rel(p), label, yes(m is not None), yes(m is not None and m.can_sft),
                      yes(m is not None and m.can_dpo))
        console.print(t)
    else:
        console.print("[bold]Models[/]  models/ is empty: copy a model folder (config.json + weights) into it.")
    apis = slash.api_configs()
    if apis:
        t = Table(box=None, header_style="bold", title="APIs", title_justify="left", title_style="bold")
        for col in ("Config", "Name", "Voice cloning", "Key"):
            t.add_column(col)
        for a in apis:
            try:
                cfg = models.detect_api(str(a)).api
                missing = apiconfig.missing_env(cfg)
                t.add_row(_rel(a), cfg.get("name", a.stem), str(cfg.get("voice_cloning", "?")),
                          f"[yellow]missing: {', '.join(missing)}[/]" if missing else "[green]found[/]")
            except SystemExit as e:
                t.add_row(_rel(a), "[red]invalid[/]", "", str(e.code)[:60])
        console.print(t)
    console.print("[dim]Other model types: docs/adding_a_model.md[/]")


AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus")


def show_voices(console):
    from rich.table import Table
    voices = slash.voice_folders()
    if not voices:
        console.print("No voices yet: put clips of one speaker in [bold]voices/<name>/[/] "
                      "(30+ minutes, 3–15 s each, at least 50 clips; see voices/README.md).")
        return
    t = Table(box=None, header_style="bold")
    for col in ("Voice", "Clips", "Minutes", "Transcripts", "Ready"):
        t.add_column(col, justify="right" if col in ("Clips", "Minutes") else "left")
    for v in voices:
        clips = [f for f in v.iterdir() if f.suffix.lower() in AUDIO_EXT]
        secs, unknown = 0.0, 0
        for f in clips:
            if f.suffix.lower() == ".wav":
                try:
                    with wave.open(str(f)) as w:
                        secs += w.getnframes() / w.getframerate()
                    continue
                except (wave.Error, EOFError, OSError):
                    pass
            unknown += 1
        minutes = f"{secs / 60:.1f}" + ("+" if unknown else "")
        meta = "metadata.csv" if (v / "metadata.csv").exists() else "made with Whisper"
        ready = "[green]yes[/]" if len(clips) >= 50 else f"[yellow]needs 50+ clips[/]"
        t.add_row(_rel(v), str(len(clips)), minutes, meta, ready)
    console.print(t)


def show_help(console, name=""):
    from rich.table import Table
    if name:
        cmd = slash.resolve(name) or slash.closest(name)
        if cmd is None:
            console.print(f"No command /{name}.")
            return
        console.print(f"[bold]/{cmd.name}[/]  {cmd.about}")
        if cmd.usage:
            console.print(f"  {cmd.usage}", highlight=False)
        sub = _subparser(cmd.name)
        if sub is not None:
            opts = _options(cmd.name)
            if opts:
                t = Table(box=None, show_header=False, pad_edge=False)
                t.add_column(style="bold")
                t.add_column()
                for o, h, _ in opts:
                    t.add_row("  " + o, h)
                console.print(t)
        if cmd.name in ("benchmark", "sft", "dpo", "setup"):
            console.print(f"[dim]Type /{cmd.name} on its own to be asked step by step.[/]")
        return
    t = Table(box=None, show_header=False, pad_edge=False)
    t.add_column(style=f"bold {ACCENT}")
    t.add_column()
    for c in COMMANDS:
        t.add_row("/" + c.name, c.about)
    console.print(t)
    console.print("[dim]/help <command> shows its options. Tab completes; Up/Down recalls earlier commands; "
                  "Ctrl+C stops a job; Ctrl+D leaves.[/]")


# ---- running a job -------------------------------------------------------------------------------

STEP = re.compile(r"^== \[\d\d:\d\d:\d\d\] (.+)")


class JobView:
    """The status line under a running job's output (rich Live; the output scrolls above it)."""

    def __init__(self, console, title):
        self.console, self.title, self.t0 = console, title, time.time()
        self.step, self.note, self.live, self.proxy = "", "", None, None

    def __rich__(self):
        from rich.spinner import Spinner
        from rich.text import Text
        el = _duration(time.time() - self.t0)
        bits = [self.title] + [b for b in (self.step, self.note) if b] + [el, "Ctrl+C stops"]
        return Spinner("dots", Text(" · ".join(bits), style="dim"), style=ACCENT)

    def start(self):
        from rich.live import Live
        self.live = Live(self, console=self.console, refresh_per_second=4, transient=True,
                         redirect_stdout=True, redirect_stderr=True)
        self.live.start()
        self.proxy = sys.stdout
        sys.stdout = _Sniff(self.proxy, self)

    def stop(self):
        if self.live:
            sys.stdout = self.proxy
            self.live.stop()
            self.live = None

    @contextmanager
    def paused(self):
        """For questions asked in the middle of a job (--review-texts)."""
        self.stop()
        try:
            yield
        finally:
            self.start()

    def saw(self, line):
        m = STEP.match(line)
        if m:
            self.step, self.note = m.group(1), ""
        elif line.startswith("turing-tester: "):
            self.note = line[len("turing-tester: "):].strip()[:70]


class _Sniff:
    """stdout wrapper that tells the status line which step is running."""

    def __init__(self, inner, view):
        self.inner, self.view, self.buf = inner, view, ""

    def write(self, s):
        self.buf += s
        *lines, self.buf = self.buf.split("\n")
        for line in lines:
            self.view.saw(line)
        return self.inner.write(s)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _duration(s):
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {s % 3600 // 60:02d}m"


def run_job(console, cmd, argv):
    from turing_tester import cli
    view = JobView(console, "/" + cmd.name)
    util.PAUSE = view.paused
    view.start()
    try:
        rc = cli.main(argv)
    except KeyboardInterrupt:
        rc = 130
    except SystemExit as e:                # argparse: bad options, or -h
        rc = e.code if isinstance(e.code, int) else 2
    except Exception:  # noqa: BLE001  (keep the interface alive; show what broke)
        view.stop()
        console.print(traceback.format_exc(), style="red", highlight=False, markup=False)
        rc = 1
    finally:
        view.stop()
        util.PAUSE = None
    took = _duration(time.time() - view.t0)
    if rc == 0:
        console.print(f"[green]✓[/] /{cmd.name} finished in {took}")
    elif rc == 130:
        console.print(f"[yellow]■[/] /{cmd.name} stopped after {took}. Run the same command again to continue "
                      "where it left off.")
    else:
        console.print(f"[red]✗[/] /{cmd.name} did not finish (exit code {rc}, after {took}).")


# ---- the prompt ----------------------------------------------------------------------------------

# Block letters for the title: solid blocks, with a box-drawn shadow down and to the right.
FONT = {
    "T": ["████████╗", "╚══██╔══╝", "   ██║   ", "   ██║   ", "   ██║   ", "   ╚═╝   "],
    "U": ["██╗   ██╗", "██║   ██║", "██║   ██║", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
    "R": ["██████╗ ", "██╔══██╗", "██████╔╝", "██╔══██╗", "██║  ██║", "╚═╝  ╚═╝"],
    "I": ["██╗", "██║", "██║", "██║", "██║", "╚═╝"],
    "N": ["███╗   ██╗", "████╗  ██║", "██╔██╗ ██║", "██║╚██╗██║", "██║ ╚████║", "╚═╝  ╚═══╝"],
    "G": [" ██████╗ ", "██╔════╝ ", "██║  ███╗", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
    "E": ["███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "███████╗", "╚══════╝"],
    "S": ["███████╗", "██╔════╝", "███████╗", "╚════██║", "███████║", "╚══════╝"],
}
LOGO_TOP, LOGO_BOTTOM, LOGO_SHADOW = (0x7f, 0xe0, 0xd8), (0x3a, 0x8f, 0xd8), "#2a5560"


def _gradient(row, rows):
    f = row / max(rows - 1, 1)
    return "#" + "".join(f"{round(a + (b - a) * f):02x}" for a, b in zip(LOGO_TOP, LOGO_BOTTOM))


def logo(width):
    """TURING / TESTER in block letters (blocks shaded top to bottom, shadow darker), or None when
    the terminal is too narrow for it."""
    from rich.text import Text
    words = ["TURING", "TESTER"]
    rows = [["".join(FONT[ch][r] for ch in w) for r in range(6)] for w in words]
    if max(len(line) for word in rows for line in word) + 2 > width:
        return None
    text = Text(no_wrap=True)
    total = 6 * len(words)
    for wi, word in enumerate(rows):
        if wi:
            text.append("\n")
        for r, line in enumerate(word):
            color = _gradient(wi * 6 + r, total)
            text.append(" ")
            for ch in line:
                text.append(ch, style=f"bold {color}" if ch == "█" else LOGO_SHADOW)
            text.append("\n")
    return text


def _banner(console):
    from rich.panel import Panel
    from rich.text import Text
    from turing_tester import hardware
    art = logo(console.width)
    if art is not None:
        console.print()
        console.print(art, end="")
    kind, gpu = hardware.accelerator()
    gpu_line = (f"{gpu['name']} · {gpu['total_gb']:.0f} GB" if kind == "cuda" else
                "Apple Silicon (MPS)" if kind == "mps" else "no NVIDIA GPU found (run /doctor)")
    n_models = len([p for p in slash.model_folders() if p.parent == paths.MODELS])
    n_apis, n_voices = len(slash.api_configs()), len(slash.voice_folders())

    def plural(n, word):
        return f"{n} {word}{'' if n == 1 else 's'}"
    body = Text()
    if art is None:
        body.append("Turing Tester\n", style=f"bold {ACCENT}")
    body.append("How close does your TTS model sound to a real person?\n\n", style="italic")
    body.append("GPU     ", style="dim")
    body.append(gpu_line + "\n")
    body.append("found   ", style="dim")
    body.append(f"{plural(n_models, 'model')} in models/ · {plural(n_apis, 'API')} in apis/ · "
                f"{plural(n_voices, 'voice')} in voices/\n\n")
    body.append("/doctor", style="bold")
    body.append(" to check this machine · ")
    body.append("/help", style="bold")
    body.append(" for all commands")
    console.print(Panel(body, border_style=ACCENT, expand=False, padding=(1, 2)))
    return gpu_line


def main():
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style
    from rich.console import Console

    os.chdir(paths.REPO)                 # the paths you type and the suggestions are relative to the repo
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and (stream.encoding or "").lower().replace("-", "") != "utf8":
            stream.reconfigure(encoding="utf-8", errors="replace")     # e.g. output piped on Windows
    console = Console(highlight=False)
    gpu_line = _banner(console)
    paths.HOME.mkdir(parents=True, exist_ok=True)
    session = PromptSession(
        history=FileHistory(str(paths.HOME / "history")), completer=_make_completer(),
        complete_while_typing=True, auto_suggest=AutoSuggestFromHistory(), key_bindings=_key_bindings(),
        style=Style.from_dict(STYLE), reserve_space_for_menu=8,
        placeholder=[("class:placeholder", "Type / for commands")],
        bottom_toolbar=lambda: f" {gpu_line}  ·  / commands  ·  Tab completes  ·  Ctrl+D leaves")
    prefill, last_ctrl_c = "", 0.0
    while True:
        try:
            line = session.prompt([("class:prompt", "> ")], default=prefill)
        except KeyboardInterrupt:
            if time.time() - last_ctrl_c < 3:
                break
            last_ctrl_c = time.time()
            console.print("[dim]Press Ctrl+C again to leave (or type /exit).[/]")
            prefill = ""
            continue
        except EOFError:
            break
        prefill = ""
        line = line.strip()
        if not line:
            continue
        if not line.startswith("/"):
            console.print("Commands start with [bold]/[/]: type [bold]/[/] to see them, or [bold]/help[/]. "
                          "[dim](Text without / is not sent anywhere.)[/]")
            continue
        try:
            words = slash.split(line)
        except ValueError as e:
            console.print(f"[red]{e}[/]")
            continue
        typed, args = words[0], words[1:]
        cmd = slash.resolve(typed)
        if cmd is None:
            guess = slash.closest(typed)
            if guess is None:
                console.print(f"Unknown command [bold]{typed}[/]. Type [bold]/help[/] to see them all.")
            else:
                console.print(f"Unknown command [bold]{typed}[/]. Did you mean [bold]/{guess.name}[/]? "
                              "[dim](Enter runs it; edit first if you like)[/]")
                rest = line.split(None, 1)[1] if " " in line else ""
                prefill = f"/{guess.name} {rest}".rstrip()
            continue
        if cmd.name == "exit":
            break
        try:
            if not dispatch(console, cmd, args):
                break
        except Cancelled:
            console.print("[dim]Cancelled.[/]")
    console.print("[dim]Bye.[/]")
    return 0


def dispatch(console, cmd, args):
    """Run one command. False = leave."""
    if cmd.name == "help":
        show_help(console, args[0] if args else "")
    elif cmd.name == "clear":
        console.clear()
    elif cmd.name == "models":
        show_models(console)
    elif cmd.name == "voices":
        show_voices(console)
    elif cmd.name == "runs":
        show_runs(console)
    elif cmd.name == "report":
        if not args:
            runs = list_runs()
            if not runs:
                console.print("No runs yet. Start one with [bold]/benchmark[/].")
                return True
            args = [select(console, "Which run?", [(r["id"], r["id"], f"{r['kind']}, {r['state']}")
                                                   for r in runs[:9]])]
        show_report(console, args[0])
    elif cmd.name == "doctor":
        from turing_tester import cli
        cli.main(["doctor"])
    else:                                  # setup, benchmark, sft, dpo: long jobs
        if not args:
            args = guided(console, cmd.name)
        try:
            argv = slash.to_argv(cmd, args)
        except ValueError as e:
            console.print(f"[red]{e}[/]")
            return True
        run_job(console, cmd, argv)
    return True

