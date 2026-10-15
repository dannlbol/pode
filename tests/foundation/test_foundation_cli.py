import subprocess
import sys

import pytest


def test_build_parser_registers_all_subcommands():
    from pode_foundation.cli import build_parser

    parser = build_parser()
    sub_actions = [a for a in parser._subparsers._group_actions if a.dest == "command"]
    choices = set(sub_actions[0].choices)
    assert choices == {"preprocess", "combine", "pretrain", "finetune", "embed", "survival"}


def test_building_parser_does_not_import_torch():
    """`pode foundation --help` must stay fast even without torch installed --
    building the argparse parser should never trigger the heavy imports that
    live inside the command handlers."""
    code = (
        "import sys\n"
        "import pode_foundation.cli as cli\n"
        "cli.build_parser()\n"
        "assert 'torch' not in sys.modules, sys.modules.keys()\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_importing_package_does_not_import_torch():
    code = (
        "import sys\n"
        "import pode_foundation\n"
        "assert 'torch' not in sys.modules, sys.modules.keys()\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_lazy_attribute_access_resolves():
    import pode_foundation as pf

    assert pf.MethylationMAE.__name__ == "MethylationMAE"
    with pytest.raises(AttributeError):
        pf.definitely_not_a_real_attribute


def test_help_exits_cleanly():
    from pode_foundation.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0
