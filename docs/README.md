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

The compiled docs are written under `docs/_build/html`.

## Read the Docs configuration

Read the Docs reads the repository-level `readthedocs.yml`.
`.readthedocs.yaml` is kept in sync for local/developer compatibility.
