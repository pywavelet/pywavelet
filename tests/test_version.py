import re

import git

import pywavelet


def test_version():
    assert hasattr(
        pywavelet, "__version__"
    ), "pywavelet has no __version__ attribute"
    assert isinstance(
        pywavelet.__version__, str
    ), f"{pywavelet.__version__} is not a string"
    assert pywavelet.__version__ > "0.0.0", f"{pywavelet.__version__} <= 0.0.0"

    # Get the latest git tag
    repo = git.Repo(search_parent_directories=True)
    git_tag = repo.git.describe(tags=True, abbrev=0)

    # Check that the version matches the last tag
    re_match = re.match(r"v(\d+\.\d+\.\d+)", git_tag)
    assert (
        re_match.group(1) == pywavelet.__version__
    ), f"{re_match.group(1)} != {pywavelet.__version__}"
