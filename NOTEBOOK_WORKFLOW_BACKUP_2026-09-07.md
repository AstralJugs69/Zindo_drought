# Kaggle workflow backup — 2026-09-07

Captured from the live `notebook6a32e6b5bf` draft before any edits or execution.
The session was off. Outputs below are saved notebook outputs only, not evidence of
a live kernel or surviving files.

## Cell 1 — original bootstrap

```python
import os
import shutil
import subprocess


repo_dir = "/kaggle/working/Zindo_drought"


# Move somewhere safe BEFORE deleting the old clone.
os.chdir("/kaggle/working")


if os.path.exists(repo_dir):
    shutil.rmtree(repo_dir)


subprocess.run(
    [
        "git",
        "clone",
        "https://github.com/AstralJugs69/Zindo_drought.git",
        repo_dir,
    ],
    check=True,
)


os.chdir(repo_dir)


commit = subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"],
    text=True,
).strip()


print("Repo ready:", os.getcwd())
print("Commit:", commit)
print("Files:", os.listdir(repo_dir))
```

Saved output: clone commit `a659e29`.

## Cell 2 — original dataset discovery

```python
import os


for root, dirs, files in os.walk("/kaggle/input"):
    if "Train.csv" in files and "Test.csv" in files:
        DATA_DIR = root
        print("Found dataset at:", DATA_DIR)
        print("Files:", sorted(files))
        break
else:
    raise FileNotFoundError("Could not locate Train.csv and Test.csv")
```

Saved output: `/kaggle/input/datasets/cashgenenator/drought`, containing Train.csv,
Test.csv, and SampleSubmission.csv.

## Cell 3 — original runner

```python
import os
import subprocess


os.chdir("/kaggle/working/Zindo_drought")


subprocess.run(["git", "pull"], check=True)


subprocess.run(
    [
        "python",
        "scripts/train_exp009_hybrid_submit.py",
        "--data-dir", "/kaggle/input/datasets/cashgenenator/drought",
    ],
    check=True,
)
```

Saved output: execution `[43]` pulled `0008f2c..a2f4e94`; it is historical only.
