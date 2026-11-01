"""Write tests/golden/translations.json: the package's output on 10 ALT test pairs.

    NETRA_NMT_BUNDLES=training/bundles python tests/make_golden.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from netra_nmt import NetraTranslator  # noqa: E402
from netra_nmt_train.data import load_pairs  # noqa: E402

pairs = load_pairs("alt", limit=10)
t = NetraTranslator(device="cpu")
cases = []
for i, (en, km) in enumerate(pairs):
    for direction, text in (("en2km", en), ("km2en", km)):
        for mode in ("greedy", "beam"):
            cases.append({"id": i, "direction": direction, "mode": mode, "input": text,
                          "output": t.translate(text, direction=direction, mode=mode)})
out = Path(__file__).parent / "golden" / "translations.json"
out.write_text(json.dumps(cases, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"{len(cases)} cases -> {out}")
