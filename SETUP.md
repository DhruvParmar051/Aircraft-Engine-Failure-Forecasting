# Setup Instructions

Follow these steps after cloning the repository to set up your environment.

## 1. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
# or: source venv/bin/activate  # On macOS/Linux
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Install nbstripout Git Hooks

```bash
nbstripout --install --attributes .gitattributes
```

This ensures that Jupyter notebooks are automatically cleaned of outputs and metadata before being committed, keeping the repository clean and reducing merge conflicts.

## Why nbstripout?

- **Removes cell outputs**: Notebooks contain large amounts of binary data (plots, dataframes) that bloat the git repo
- **Removes metadata**: Kernel info, execution times, etc. are not needed for version control
- **Reduces merge conflicts**: With outputs stripped, two people editing the same notebook won't have conflicts from different outputs
- **Automatic cleaning**: Works transparently with `git add` and `git commit`

## Verification

After running setup, verify nbstripout is installed:

```bash
git config filter.nbstripout.clean
# Should output: nbstripout --strip-empty-cells
```

## Next Steps

1. Pull the latest code from the **main** branch
2. All preprocessing and features are completed (T01, T02, T03)
3. Processed data is available in `data/processed/`
4. Use the shared utilities:
   - `src.features.windowing.create_windows()` — sliding window sequences
   - `src.evaluation.metrics.rmse()` and `metrics.nasa_score()` — evaluation

See [README.md](README.md) for full pipeline documentation.
