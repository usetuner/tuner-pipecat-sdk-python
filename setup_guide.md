Create a venv with **Python 3.12 or 3.13** (not the system 3.14 if that is your default):

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # macOS/Linux
# or: uv venv --python 3.12 && source .venv/bin/activate
```

**Into that venv:**

```bash
pip install -e .
# optional dev/test extras
pip install -e ".[dev]"
```

**Examples** under `examples/` depend on this SDK via a **local path** (`tool.uv.sources`). Use **`uv sync`** inside each example directory, or install the SDK from the repo root with `pip install -e .` first, then `pip install -e .` in the example (plain `pip` does not read `uv.sources`).


## Examples

See [`examples/`](examples/) for working bots built with this SDK — the full list with
setup instructions lives in [`examples/README.md`](examples/README.md).

Each example is self-contained. See **Installation** above for how the SDK is resolved. To run one:

```bash
cd examples/<example_name>
uv sync
cp .env.example .env   # if present; fill in API keys
uv run <script>.py
```

Then open http://localhost:7860 where applicable.

## Publish to PyPI

Releases are automated. `.github/workflows/publish.yml` runs on every push to
`main` (so on every merged PR) and publishes only when the version in
`pyproject.toml` is not yet on PyPI. Ordinary merges are a no-op; a merge that
bumps the version runs the tests, builds, uploads, tags `vX.Y.Z` and opens a
GitHub release.

### Releasing

1. In the PR, bump `version` in `pyproject.toml` and move the `Unreleased`
   section of `CHANGELOG.md` under the new version.
2. Merge the PR. The Publish workflow does the rest.

### One-time setup

PyPI trusted publishing (OIDC) — no API token is stored anywhere:

1. On PyPI, sign in as a project owner and go to
   [Your projects](https://pypi.org/manage/projects/) → *Manage* on
   `tuner-pipecat-sdk` → *Publishing* in the sidebar.
2. Under the *GitHub Actions* tab, fill in and press *Add*:

   | Field            | Value                      |
   | ---------------- | -------------------------- |
   | Repository owner | `usetuner`                 |
   | Repository name  | `tuner-pipecat-sdk-python` |
   | Workflow name    | `publish.yml`              |
   | Environment name | `pypi`                     |

   The workflow field takes the bare filename, not the `.github/workflows/`
   path. The environment field is optional on PyPI but must be filled here,
   because `publish.yml` runs in the `pypi` environment.
3. On GitHub, create the matching environment: *Settings → Environments → New
   environment* → `pypi`, then under *Deployment branches and tags* choose
   *Selected branches and tags* and add a rule for `main`. PyPI's publisher has
   no branch field, so this is the only place a branch restriction can live.
   Add required reviewers too if a release should wait for a manual approval —
   and leave *Allow administrators to bypass configured protection rules*
   unchecked, or the gate is advisory.

Nothing is stored in GitHub secrets. If the first run fails with
`invalid-publisher`, the cause is almost always a mismatch between these four
values and the workflow — renaming `publish.yml` or the `pypi` environment
means editing the publisher on PyPI too.

### Manual release (fallback)

```bash
uv sync --extra dev
rm -rf dist/
uv build
uv run twine check dist/*
uv run twine upload --repository testpypi dist/*   # optional smoke test
uv run twine upload dist/*
```
