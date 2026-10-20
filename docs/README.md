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

## How the docs are organized

The docs have five sections, each a directory with an `index.rst`: `overview/` (what sparrow is, how it is organized, installation), `protein/` (a page for each kind of thing a `Protein` gives), `batch/` (`sparrow.batch`, and every analysis it accepts), `power_user/` (the functions underneath) and `api/` (every module, from its docstring). Five pages sit beside them at the top level: the worked examples (`examples.rst`), the guides to adding a predictor and writing a plugin, how to cite sparrow, and the changelog.

Each object is documented in full in one place, its *canonical* entry, which every cross-reference links to. A member of `Protein` or of its `predictor`, `polymeric` and `nardini` accessors has its canonical entry on its page in `protein/`; everything else has its canonical entry in `api/`. Where the API reference shows a class whose members live in `protein/`, it does so with `:no-index:`, so the two do not collide. An `autoclass` or `automodule` that should not document its members needs `:no-inherited-members:` as well as `:no-members:`, because `conf.py` turns `inherited-members` on by default and that alone documents every member.

## The docs' own extensions

Three small Sphinx extensions live in `docs/_ext/`:

- `batch_analyses.py` writes the reference for every name `sparrow.batch` accepts (`docs/batch/_analyses_*.rst`, which the pages in `docs/batch/` include) on every build, from the same code `batch` uses to check a name, so it cannot fall out of step with it.
- `docs_coverage.py` warns, and so fails the strict build, if a public property or method of `Protein` or its accessors has no entry on any page, or a public module has no page in `api/`. When you add one, give it an entry; a module that should not be documented goes in its `NOT_DOCUMENTED`, with the reason.
- `redirects.py` writes a page at the old address of each page that moved when the docs were reorganized, sending the reader on to the new one.

## Read the Docs configuration

Read the Docs reads the repository-level `readthedocs.yml`.
`.readthedocs.yaml` is kept in sync for local/developer compatibility.
