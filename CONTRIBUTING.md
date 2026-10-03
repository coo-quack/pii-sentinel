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

## Training and Evaluation

```bash
uv run python -m pii_sentinel.gen.generator --out data --train-docs 2200 --templates templates \
  --template-weight 110 \
  --exclude "$(ls eval/*.json eval/reference/*.json eval/work/raw/*.json | paste -sd, -)"
uv run python -m pii_sentinel.train --data data --out models/mmBERT-pii-sentinel
uv run python -m pii_sentinel.evaluate --model models/mmBERT-pii-sentinel eval/dev.json eval/test.json --out runs
# --no-rules scores the model alone, without the rule set
```

Training data is generated from templates; labels come from what each template planted, not from a model.
Names, brands and public figures that appear in the evaluation corpora are excluded from generation.

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

When a newly trained model is adopted, upload it to the `develop` branch of the Hub repository and record its
hashes (the weights are not kept in Git):

```bash
(cd models/mmBERT-pii-sentinel && shasum -a 256 config.json model.safetensors document_heads.safetensors \
  pii_sentinel.json tokenizer.json tokenizer_config.json) > model.sha256
hf upload coo-quack/mmBERT-pii-sentinel models/mmBERT-pii-sentinel . --revision develop \
  $(awk '{printf " --include %s", $2}' model.sha256)
```

When bumping a version, open a PR from `develop` → `main` with:

1. Update `version` in `pyproject.toml`
2. Add a `## vX.Y.Z (YYYY-MM-DD)` section to `CHANGELOG.md`
   - `release.yml` extracts that section as the GitHub Release notes, so the heading must start with `## vX.Y.Z`
3. Update the evaluation table in `MODEL_CARD.md` when the model or the post-processing changed
4. Update the version in the `uvx --from git+...@vX.Y.Z` commands of `README.md` and `MODEL_CARD.md`

After merging into `main`, `release.yml` automatically:

- Downloads the model from the Hub's `develop` branch and checks it against `model.sha256`
- Uploads it with `MODEL_CARD.md` (as the Hub README), `LICENSE` and `THIRD_PARTY_NOTICES.md` to the Hub's `main`
  branch and tags it `vX.Y.Z`
- Creates the git tag `vX.Y.Z` and a GitHub Release with the notes from `CHANGELOG.md`

It needs the repository secret `HF_TOKEN`, a Hugging Face token with write access to the model repository.

## Pull Requests

- Feature PRs target `develop`, release/hotfix PRs target `main`
- Follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `ci:`, `chore:`)
- Keep the CI job names (`audit`, `lint`, `test`) unchanged: they are required status checks managed in coo-quack/iac

## Data and Evaluation

- Training data, templates and test fixtures contain synthetic personal information only; never add real names,
  addresses or numbers.
- Gold labels follow `docs/labeling-policy.md`. Never change a label to match the model's output.
- The test half of the evaluation set (`eval/test.json`) is not committed and is used for aggregate scores only.
