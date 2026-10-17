import os
from pathlib import Path
from typing import Any
import orjson
from dotenv import load_dotenv

load_dotenv()

ENV = os.environ.get("PEEK_ENV", "").strip().lower()
IS_KAGGLE = (ENV == "kaggle") or Path("/kaggle/working").exists()

REPO_ROOT = Path(__file__).resolve().parent.parent

if IS_KAGGLE:
    DATA_ROOT = Path("/kaggle/working/data")
    CKPT_ROOT = Path("/kaggle/working/checkpoints")
    RESULTS_ROOT = Path("/kaggle/working/results")
    HF_CACHE = Path("/kaggle/working/.cache/huggingface")
else:
    DATA_ROOT = REPO_ROOT / "data"
    CKPT_ROOT = REPO_ROOT / "checkpoints"
    RESULTS_ROOT = REPO_ROOT / "results"
    HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))

for d in [DATA_ROOT, CKPT_ROOT, RESULTS_ROOT]:
    d.mkdir(parents=True, exist_ok=True)


def load_json(filepath: Path | str) -> Any:
    """Read json file using orjson."""
    with open(filepath, "rb") as f:
        return orjson.loads(f.read())


def save_json(data: Any, filepath: Path | str) -> None:
    """Write json file using orjson."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))


def save_jsonl(records: list[dict], filepath: Path | str) -> None:
    """Write JSONL file using orjson."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for r in records:
            f.write(orjson.dumps(r) + b"\n")


def load_jsonl(filepath: Path | str) -> list[dict]:
    """Read JSONL file using orjson."""
    records = []
    with open(filepath, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(orjson.loads(line))
    return records
