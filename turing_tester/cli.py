"""turing-tester: benchmark, SFT and DPO for TTS models (see README.md)."""
import argparse
import sys

from turing_tester.util import Fail

EPILOG = """examples:
  ./turing-tester doctor
  ./turing-tester benchmark --model models/my-tts
  ./turing-tester benchmark --api apis/elevenlabs.yaml --limit 10
  ./turing-tester sft --model models/my-tts --teacher models/big-tts
  ./turing-tester sft --model models/my-tts --voice voices/my-voice
  ./turing-tester dpo --model runs/sft/my-tts/final
(on Windows: .\\turing-tester ...)"""


def build_parser():
    p = argparse.ArgumentParser(prog="turing-tester", description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", metavar="command")

    sub.add_parser("doctor", help="check this machine: GPU, memory, disk, what is installed")

    s = sub.add_parser("setup", help="install environments now instead of on first use")
    s.add_argument("envs", nargs="*", help="tts judge moss lm qwen3-train (default: tts judge)")

    b = sub.add_parser("benchmark", help="score a model or API against real human recordings")
    b.add_argument("--model", help="model folder in models/ (or a Hugging Face id)")
    b.add_argument("--api", help="API config, e.g. apis/elevenlabs.yaml")
    b.add_argument("--set", default="ears", help="ears (default) | libritts-r | voices/<name>")
    b.add_argument("--judges", default="grm,btrm", help="grm,btrm (default) | grm | btrm")
    b.add_argument("--limit", type=int, default=0, help="first N sentences only (quick test)")
    b.add_argument("--name", default="", help="run folder name under runs/ (default: from the model)")
    b.add_argument("--votes", type=int, default=1, help=argparse.SUPPRESS)
    b.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)

    t = sub.add_parser("sft", help="fine-tune a model on good examples")
    t.add_argument("--model", required=True, help="the model to train (a folder in models/)")
    t.add_argument("--voice", help="your recordings, voices/<name>: train on your own voice")
    t.add_argument("--teacher", help="a bigger model folder or an API config that speaks the training sentences")
    t.add_argument("--ears", action="store_true",
                   help="also train on the EARS human recordings (the default when neither --voice nor --teacher is given)")
    t.add_argument("--texts", help="your own training sentences, one per line (used by the teacher)")
    t.add_argument("--review-texts", action="store_true",
                   help="pause after the diagnosis so you can review or add training sentences")
    t.add_argument("--mode", choices=("quick", "full"), default="quick",
                   help="quick: one training run with our settings; full: the complete experiment procedure")
    t.add_argument("--name", default="", help="run folder name under runs/sft/ (default: from the model)")
    t.add_argument("--n-texts", type=int, default=0, help=argparse.SUPPRESS)
    t.add_argument("--skip-final", action="store_true", help="skip the before/after benchmark")
    t.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)

    d = sub.add_parser("dpo", help="improve a model on its own best and worst samples")
    d.add_argument("--model", required=True, help="the model to improve, e.g. runs/sft/<name>/final")
    d.add_argument("--voice", help="clone your voice (voices/<name>) during DPO instead of the EARS speakers")
    d.add_argument("--texts", help="your own sentences for the rollouts, one per line")
    d.add_argument("--rounds", type=int, default=3, help="maximum rounds (default 3, as in our experiments)")
    d.add_argument("--name", default="", help="run folder name under runs/dpo/ (default: from the model)")
    d.add_argument("--skip-final", action="store_true", help="skip the before/after benchmark")
    d.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)
    return p


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    if not args.command:
        p.print_help()
        return 0
    try:
        if args.command == "doctor":
            from turing_tester import doctor
            return doctor.main(args)
        if args.command == "setup":
            from turing_tester import envs
            for name in args.envs or ["tts", "judge"]:
                envs.ensure(name)
            print("turing-tester: environments ready")
            return 0
        if args.command == "benchmark":
            from turing_tester import bench
            return bench.main(args)
        if args.command == "sft":
            from turing_tester import sft
            return sft.main(args)
        if args.command == "dpo":
            from turing_tester import dpo
            return dpo.main(args)
    except KeyboardInterrupt:
        print("\nturing-tester: stopped. Run the same command again to continue where it left off.", file=sys.stderr)
        return 130
    except Fail as e:
        print(str(e.code), file=sys.stderr)
        return 2
    return 0
