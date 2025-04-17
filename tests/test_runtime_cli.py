import os

from click.testing import CliRunner

from pywavelet.utils.timing_cli.cli import cli_collect_runtime


def test_pywavelet_timer(outdir):
    out = f"{outdir}/test_pywavelet_timer"

    runner = CliRunner()
    result = runner.invoke(
        cli_collect_runtime,
        [
            "--outdir",
            out,
            "--log2n",
            "5",
            "--nrep",
            "2",
            "--backend",
            "numpy64",
        ],
    )
    assert result.exit_code == 0, result.output

    # Check that CSV output files were generated.
    files = os.listdir(out)
    csv_files = [f for f in files if f.endswith(".csv")]
    assert len(csv_files) > 0
