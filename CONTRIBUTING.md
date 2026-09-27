# Contributing

Thanks for your interest in contributing to pii-sentinel!

## Development Setup

```bash
git clone https://github.com/coo-quack/pii-sentinel.git
cd pii-sentinel
uv sync
```

The first run downloads the mmBERT-base tokenizer and configuration from Hugging Face. Training needs a GPU;
everything else runs on a laptop (Apple silicon uses MPS).

## Commands

```bash
uv run pytest                                   # Unit tests
uv run ruff check src tests                     # Lint
uv run ruff format src tests                    # Format (write)
uv run ruff format --check src tests            # Format (check only)
uv run pii-sentinel scan --model <model> FILE   # Run the CLI
uv run python -m pii_sentinel.evaluate --model <model> eval/dev.json   # Score on the development set
```

The three required CI checks run pip-audit (`audit`), ruff (`lint`) and pytest (`test`).

## Branching Strategy

```
main
 ├── develop          ← integration branch
 │    └── <type>/*   ← everything that is not an urgent production fix
 └── hotfix/*        ← urgent production fixes
```

### Normal development

```
<type>/your-change  →  develop  →  main (release)
```

1. Branch from `develop`: `git checkout -b feat/your-change develop`
2. Open a PR targeting `develop`
3. After CI passes, merge into `develop`
4. When ready to release, open a PR from `develop` → `main`

### Hotfix

1. Branch from `main`: `git checkout -b hotfix/fix-description main`
2. Apply the fix and open a PR targeting `main`
3. After review and approval, merge into `main`
4. `backport.yml` opens the sync PR into `develop` automatically

If the sync PR has conflicts, resolve them manually before merging.

## Release Checklist

When bumping a version, open a PR from `develop` → `main` with:

1. Update `version` in `pyproject.toml`
2. Add a `## vX.Y.Z (YYYY-MM-DD)` section to `CHANGELOG.md`
3. Update the evaluation table in `MODEL_CARD.md` when the model or the post-processing changed

## Pull Requests

- Feature PRs target `develop`, release/hotfix PRs target `main`
- Follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `ci:`, `chore:`)
- Keep the CI job names (`audit`, `lint`, `test`) unchanged: they are required status checks managed in coo-quack/iac

## Data and Evaluation

- Training data, templates and test fixtures contain synthetic personal information only; never add real names,
  addresses or numbers.
- Gold labels follow `docs/labeling-policy.md`. Never change a label to match the model's output.
- The test half of the evaluation set (`eval/test.json`) is not committed and is used for aggregate scores only.
