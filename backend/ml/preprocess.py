"""
shared preprocessing: raw survey row -> numeric np.float32[22]

usage:
    from preprocess import Encoder
    enc = Encoder(schema_path='../shared/feature_schema.json')
    x = enc.transform_row(raw_dict)   # returns np.ndarray shape (22,)
"""

import json, numpy as np
from pathlib import Path
from typing import Dict, Any, List

class Encoder:
    def __init__(self, schema_path: str | Path):
        with open(schema_path, "r") as f:
            self.schema = json.load(f)

        # Build lookup tables once for speed
        self.feature_specs: List[Dict[str, Any]] = self.schema["features"]
        self.order: List[str] = self.schema["order"]
        self._spec_by_name = {f["name"]: f for f in self.feature_specs}

    def _encode_value(self, spec: Dict[str, Any], raw_value: Any) -> float:
        ftype = spec["type"]

        if ftype == "numeric":
            try:
                return float(raw_value)
            except Exception:
                return 0.0

        mapping = spec["mapping"]
        # Normalise missing / unseen
        return float(mapping.get(raw_value, mapping.get("Did not answer", 0)))

    def transform_row(self, row: Dict[str, Any]) -> np.ndarray:
        encoded: list[float] = []
        for name in self.order:
            spec = self._spec_by_name[name]
            encoded.append(self._encode_value(spec, row.get(name)))
        return np.asarray(encoded, dtype=np.float32)

# quick CLI test
if __name__ == "__main__":
    import pandas as pd, sys
    enc = Encoder(Path(__file__).resolve().parents[1] / "shared/feature_schema.json")
    # one-line demo: read first row from CSV passed in argv[1]
    if len(sys.argv) > 1:
        df = pd.read_csv(sys.argv[1])
        print(enc.transform_row(df.iloc[0].to_dict()))
    else:
        # minimal smoke test with empty dict
        print(enc.transform_row({}))    # should print 22 zeros