import os
import shutil
import tempfile

import nbformat
import pytest
from nbclient import NotebookClient

HERE = os.path.dirname(__file__)
DOCS_DIR = os.path.abspath(os.path.join(HERE, "..", "docs/examples"))

notebooks = [
    "basic.ipynb",
]


@pytest.mark.parametrize("notebook_path", notebooks)
def test_notebook_execution(notebook_path, tmp_path):
    # Create a temp folder for notebook execution
    out = tmp_path / "notebooks"
    out.mkdir(parents=True, exist_ok=True)

    temp_notebook_path = out / os.path.basename(notebook_path)
    shutil.copy(DOCS_DIR + "/" + notebook_path, temp_notebook_path)

    # Load the notebook
    with open(temp_notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Create a NotebookClient and execute -- should run in max 20 seconds
    client = NotebookClient(nb, timeout=20, kernel_name="python3")

    try:
        client.execute()
    except Exception as e:
        pytest.fail(f"Execution of notebook {notebook_path} failed: {e}")
