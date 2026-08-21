# How to contribute

We welcome contributions from external contributors, and this document
describes how to merge code changes into this sparrow. 

## Getting Started

* Make sure you have a [GitHub account](https://github.com/signup/free).
* [Fork](https://help.github.com/articles/fork-a-repo/) this repository on GitHub.
* On your local machine,
  [clone](https://help.github.com/articles/cloning-a-repository/) your fork of
  the repository.

## Setting up

sparrow is pure Python, so there is no build step. From your clone:

```bash
pip install -e '.[dev]'    # sparrow, its test dependencies, and pre-commit
pre-commit install         # once per clone - installs the git hook
```

`pre-commit install` is what wires the hooks into `git commit`; without it they
never run locally and you will only find out in CI.

**What runs when you commit:** `ruff check --fix` and `ruff format`, on the files
you staged. Both are pinned in `.pre-commit-config.yaml`, so everyone gets the
same result and no separate ruff install is needed — pre-commit fetches it.

**Type checking is not a commit gate.** Run it when you want it:

```bash
pre-commit run ty --hook-stage manual    # requires uv on PATH
```

It is manual for a reason. `ty` resolves types from your *installed* environment,
so if a dependency is missing its imports become `Unknown` and real errors stop
being reported rather than being flagged — the run passes while checking almost
nothing. CI installs the full dependency set before running it, which is the only
place that guarantee holds. Treat the diagnostic count as environment-specific,
not as a shared number.

If you do not have [uv](https://docs.astral.sh/uv/) on PATH the ty hook reports
``Executable `uvx` not found``. Since it is manual-stage, that never blocks a
commit — it only matters if you asked for it. Note that uv does not bundle ty;
`uvx` fetches and caches it on first use.

## Running the tests

```bash
pytest sparrow/tests
```

`localcider` is part of the `test` extra and is required: two test modules import
it, one of them at module scope, so without it collection fails and no tests run.

## Making Changes

* Add some really awesome code to your local fork.  It's usually a [good
  idea](http://blog.jasonmeridth.com/posts/do-not-issue-pull-requests-from-your-master-branch/)
  to make changes on a
  [branch](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/)
  with the branch name relating to the feature you are going to add.
* When you are ready for others to examine and comment on your new feature,
  navigate to your fork of sparrow on GitHub and open a [pull
  request](https://help.github.com/articles/using-pull-requests/) (PR). Note that
  after you launch a PR from one of your fork's branches, all
  subsequent commits to that branch will be added to the open pull request
  automatically.  Each commit added to the PR will be validated for
  mergability, compilation and test suite compliance; the results of these tests
  will be visible on the PR page.
* If you're providing a new feature, you must add test cases and documentation.
* When the code is ready to go, make sure you run the test suite using pytest.
* When you're ready to be considered for merging, check the "Ready to go"
  box on the PR page to let the sparrow devs know that the changes are complete.
  The code will not be merged until this box is checked, the continuous
  integration returns checkmarks,
  and multiple core developers give "Approved" reviews.

# Additional Resources

* [General GitHub documentation](https://help.github.com/)
* [PR best practices](http://codeinthehole.com/writing/pull-requests-and-other-good-practices-for-teams-using-github/)
* [A guide to contributing to software packages](http://www.contribution-guide.org)
* [Thinkful PR example](http://www.thinkful.com/learn/github-pull-request-tutorial/#Time-to-Submit-Your-First-PR)
