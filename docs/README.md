# Compiling sparrow's Documentation

The docs for this project are built with [Sphinx](http://www.sphinx-doc.org/en/master/).

## Local build

Install docs dependencies:

```bash
python -m pip install -r docs/requirements.txt
```

Build HTML docs:

```bash
make -C docs html
```

Run the strict build used for docs QA:

```bash
SPARROW_DOC_BUILD=1 python -m sphinx -n -W --keep-going -b html docs docs/_build/html
```

It fails on any warning, including every cross-reference that does not resolve. The type words of numpy-style docstrings ("optional", "sequence of str", ...) and a few third-party types are not links, and `nitpick_ignore_regex` in `docs/conf.py` lists them; anything else it reports is a broken link to fix.

To check what Read the Docs will build, which installs only `docs/requirements.txt` and mocks sparrow's heavier dependencies, build from a fresh environment with only those requirements and `READTHEDOCS=True` set: a module that fails to import there is documented as empty.

The compiled docs are written under `docs/_build/html`. They are versioned from the git tags by versioningit, as sparrow itself is.

## Read the Docs configuration

Read the Docs reads the repository-level `readthedocs.yml`.
`.readthedocs.yaml` is kept in sync for local/developer compatibility.
