"""Build-time check that the (pruned) image still runs everything the app does.

It runs offline against the baked-in model, so a prune that goes too far, or
a model that didn't make it into the image, fails the build instead of the
first request. It also checks that the heavy libraries stay out of the image.
"""

import importlib.util

for module in ("torch", "transformers", "cv2", "scipy"):
    assert importlib.util.find_spec(module) is None, f"{module} is installed in the image"

from netra_nmt import NetraTranslator  # noqa: E402

t = NetraTranslator(device="cpu")
for text, direction, mode in (("Hello, how are you?", "en2km", "greedy"),
                              ("I love Cambodia.", "en2km", "beam"),
                              ("ខ្ញុំស្រលាញ់ប្រទេសកម្ពុជា។", "km2en", "greedy")):
    out = t.translate(text, direction=direction, mode=mode)
    assert out.strip(), f"no output for {direction} {mode}"
    print(f"{direction} {mode}: {text} -> {out}")

from fastapi.testclient import TestClient  # noqa: E402

from netra_nmt.server.app import app  # noqa: E402

with TestClient(app) as client:
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/api/translate", json={"text": "Good morning", "direction": "en2km"})
    assert r.status_code == 200 and r.json()["translation"], r.text
    assert client.get("/").status_code == 200
print("smoke test passed")
