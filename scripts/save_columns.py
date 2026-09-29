from __future__ import annotations

import json

from src.data import load_dedup_split
from src.paths import ARTIFACTS


COLUMNS_PATH = ARTIFACTS / "columns.json"


def main() -> None:
    X_train, _, _, _, _ = load_dedup_split()
    columns = X_train.columns.tolist()

    COLUMNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    COLUMNS_PATH.write_text(json.dumps(columns, indent=2))

    print(f"wrote {len(columns)} columns to {COLUMNS_PATH}")


if __name__ == "__main__":
    main()
