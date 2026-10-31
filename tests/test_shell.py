"""Tests for the interactive interface: / command ranking and parsing (standard library only), and,
when prompt_toolkit and rich are installed, the suggestions, a scripted session and the job runner.

  python -m unittest discover -s tests -v
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from turing_tester import paths, slash, util  # noqa: E402

try:
    import prompt_toolkit  # noqa: F401
    import rich  # noqa: F401
    HAVE_UI = True
except ImportError:
    HAVE_UI = False


def names(cmds):
    return [c.name for c in cmds]


class Ranking(unittest.TestCase):
    def test_closest_first(self):
        self.assertEqual(names(slash.rank_commands("r"))[:2], ["report", "runs"])     # starts with
        self.assertEqual(names(slash.rank_commands("re")), ["report"])
        self.assertEqual(names(slash.rank_commands("bnch")), ["benchmark"])            # letters in order
        self.assertEqual(names(slash.rank_commands("benchmrk")), ["benchmark"])        # a typo
        self.assertEqual(names(slash.rank_commands("stup")), ["setup"])
        self.assertEqual(slash.rank_commands("xyz"), [])

    def test_everything_for_a_bare_slash(self):
        self.assertEqual(names(slash.rank_commands("")), names(slash.COMMANDS))
        self.assertEqual(names(slash.rank_commands("/")), names(slash.COMMANDS))

    def test_tiers_order(self):
        # "o": starts with nothing, contained in doctor/models/voices/report/...; all tier 1, alphabetical
        ranked = names(slash.rank_commands("o"))
        self.assertEqual(ranked, sorted(ranked))

    def test_resolve(self):
        self.assertEqual(slash.resolve("/benchmark").name, "benchmark")
        self.assertEqual(slash.resolve("bench").name, "benchmark")       # alias
        self.assertEqual(slash.resolve("doc").name, "doctor")            # the only command starting so
        self.assertEqual(slash.resolve("q").name, "exit")
        self.assertIsNone(slash.resolve("s"))                            # setup and sft
        self.assertIsNone(slash.resolve("hlp"))
        self.assertEqual(slash.closest("hlp").name, "help")


class Parsing(unittest.TestCase):
    def test_first_word_is_the_model_or_api(self):
        b, s = slash.BY_NAME["benchmark"], slash.BY_NAME["sft"]
        self.assertEqual(slash.to_argv(b, ["models/x", "--limit", "10"]),
                         ["benchmark", "--model", "models/x", "--limit", "10"])
        self.assertEqual(slash.to_argv(b, ["apis/e.yaml"]), ["benchmark", "--api", "apis/e.yaml"])
        self.assertEqual(slash.to_argv(b, ["--model", "models/x"]), ["benchmark", "--model", "models/x"])
        self.assertEqual(slash.to_argv(s, ["models/x", "--voice", "voices/v"]),
                         ["sft", "--model", "models/x", "--voice", "voices/v"])
        with self.assertRaises(ValueError):
            slash.to_argv(s, ["apis/e.yaml"])
        self.assertEqual(slash.to_argv(slash.BY_NAME["setup"], ["tts"]), ["setup", "tts"])

    def test_split(self):
        self.assertEqual(slash.split(r"/sft models\my-tts --voice 'voices/my voice'"),
                         ["/sft", "models/my-tts", "--voice", "voices/my voice"])
        with self.assertRaises(ValueError):
            slash.split("/sft 'unclosed")

    def test_path_candidates(self):
        self.assertIn("apis/", slash.path_candidates(""))
        self.assertEqual(slash.path_candidates("ap"), ["apis/"])
        self.assertIn("apis/elevenlabs.yaml", slash.path_candidates("apis/e"))
        self.assertEqual(slash.path_candidates("no-such-dir/x"), [])


@unittest.skipUnless(HAVE_UI, "needs prompt_toolkit and rich (the launcher installs them)")
class Interface(unittest.TestCase):
    def completions(self, text):
        from prompt_toolkit.document import Document
        from turing_tester import shell
        return [(c.text, c.start_position) for c in shell._make_completer().get_completions(Document(text), None)]

    def test_suggestions(self):
        self.assertEqual(self.completions("/r")[:2], [("report ", -1), ("runs ", -1)])
        self.assertEqual(self.completions("/bnch")[0], ("benchmark ", -4))
        self.assertEqual(self.completions("hello"), [])
        opts = [t for t, _ in self.completions("/benchmark --l")]
        self.assertEqual(opts[0], "--limit")
        self.assertIn("ears", [t for t, _ in self.completions("/benchmark x --set ")])
        self.assertIn("apis/elevenlabs.yaml", [t for t, _ in self.completions("/benchmark ap")] +
                      [t for t, _ in self.completions("/benchmark apis/")])
        self.assertIn("judge", [t for t, _ in self.completions("/setup tts ")])

    def test_logo(self):
        from turing_tester import shell
        art = shell.logo(100)
        lines = art.plain.splitlines()
        self.assertEqual(len(lines), 13)                       # two words of 6 rows and a gap
        self.assertTrue(all(len(line) <= 52 for line in lines))
        self.assertIsNone(shell.logo(40))                      # too narrow: plain title instead

    def run_session(self, keys):
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput
        from turing_tester import shell
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as home, create_pipe_input() as inp, \
                mock.patch.object(paths, "HOME", Path(home)), mock.patch.object(shell, "_banner", lambda c: "gpu"), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            inp.send_text(keys)
            with create_app_session(input=inp, output=DummyOutput()):
                rc = shell.main()
        os.chdir(REPO)
        return rc, out.getvalue()

    def test_session(self):
        rc, out = self.run_session("/hlp\r\r/help sft\rhello\r/benchmrk models/nope\r\r/exit\r")
        self.assertEqual(rc, 0)
        self.assertIn("Did you mean /help?", out)
        self.assertIn("/voices", out)                                   # the accepted /help listing
        self.assertIn("--teacher", out)                                 # /help sft
        self.assertIn("Commands start with /", out)                     # text without /
        self.assertIn("model folder not found: models/nope", out)       # /benchmark ran after the fix
        self.assertIn("did not finish", out)

    def test_guided_cancel(self):
        rc, out = self.run_session("/benchmark\r1\r\x1b[B\r2/exit\r")    # first model, EARS, full set, Cancel
        self.assertIn("Same as typing: /benchmark", out)
        self.assertIn("Cancelled.", out)

    def test_job_runner(self):
        from rich.console import Console
        from turing_tester import cli, shell

        def fake(argv):
            print("== [12:00:00] generate")
            with util.PAUSE():
                pass
            raise KeyboardInterrupt

        out = io.StringIO()
        before = sys.stdout
        with mock.patch.object(cli, "main", fake), contextlib.redirect_stdout(io.StringIO()):
            shell.run_job(Console(file=out), slash.BY_NAME["benchmark"], ["benchmark"])
        self.assertIs(sys.stdout, before)
        self.assertIsNone(util.PAUSE)
        self.assertIn("stopped after", out.getvalue())


if __name__ == "__main__":
    unittest.main()
