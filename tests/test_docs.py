import os
import shutil
import tempfile

import nbformat
import pytest
from nbconvert import PythonExporter

HERE = os.path.dirname(__file__)
DOCS_DIR = os.path.abspath(os.path.join(HERE, "..", "docs"))

# List of notebooks to test
notebooks = [
    "example.ipynb",
]


@pytest.mark.parametrize("notebook_path", notebooks)
def test_notebook_execution(notebook_path, outdir):
    # Create a dir for the notebook tests
    out = os.path.join(outdir, "notebooks")
    os.makedirs(out, exist_ok=True)

    # Copy the notebook to the temporary directory
    temp_notebook_path = os.path.join(out, os.path.basename(notebook_path))
    shutil.copy(f"{DOCS_DIR}/{notebook_path}", temp_notebook_path)

    # Change the working directory to the notebooks dir
    os.chdir(os.path.dirname(out))

    # Load the notebook
    with open(temp_notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Convert the notebook to a Python script
    exporter = PythonExporter()
    source, _ = exporter.from_notebook_node(nb)

    # Define the execution environment
    exec_env = {}

    # Execute the Python script
    try:
        exec(source, exec_env)
    except Exception as e:
        pytest.fail(f"Execution of notebook {notebook_path} failed: {e}")
