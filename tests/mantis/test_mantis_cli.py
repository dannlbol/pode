import subprocess
import sys

import pytest


def test_build_parser_registers_all_subcommands():
    from pode_mantis.cli import build_parser

    parser = build_parser()
    sub_actions = [a for a in parser._subparsers._group_actions if a.dest == "command"]
    choices = set(sub_actions[0].choices)
    assert choices == {
        "prepare-beta", "prepare-cnv", "prepare-survival", "generate-config", "infer", "evaluate", "plot",
    }


def test_building_parser_does_not_import_pandas():
    """`pode mantis --help` should stay fast -- building the argparse parser
    should never trigger the handlers' heavy imports (pandas/yaml/subprocess
    logic all live inside the command functions, not at module scope)."""
    code = (
        "import sys\n"
        "import pode_mantis.cli as cli\n"
        "cli.build_parser()\n"
        "assert 'pandas' not in sys.modules, sys.modules.keys()\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_importing_package_does_not_import_pandas():
    code = (
        "import sys\n"
        "import pode_mantis\n"
        "assert 'pandas' not in sys.modules, sys.modules.keys()\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_lazy_attribute_access_resolves():
    import pode_mantis as pm

    assert pm.run_prepare_beta.__name__ == "run_prepare_beta"
    with pytest.raises(AttributeError):
        pm.definitely_not_a_real_attribute


def test_help_exits_cleanly():
    from pode_mantis.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0
