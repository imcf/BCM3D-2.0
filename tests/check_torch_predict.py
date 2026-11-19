"""Check that the PyTorch port matches CSBDeep/TensorFlow.

Usage: python tests/check_torch_predict.py <model_dir>
(<model_dir> holds weights_best.h5 and config.json, e.g. from OSF m4637).
"""
import sys
from pathlib import Path

import numpy as np
from csbdeep.models import CARE

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.torch_predict import load_model, predict  # noqa: E402

model_dir = Path(sys.argv[1])
rng = np.random.default_rng(0)
vol = rng.random((30, 70, 66), np.float32)  # odd sizes test the padding
vol[:, 20:50, 10:40] += 2  # some structure
care = CARE(config=None, name=model_dir.name, basedir=str(model_dir.parent))
ref = care.predict(np.stack((vol, vol), 1), "ZCYX", n_tiles=(1, 1, 1, 1))
out = predict(load_model(model_dir, "cpu"), vol, tile=32)  # tiled
err = np.abs(out - ref).max() / (ref.max() - ref.min())
assert err < 1e-3, err
print(f"ok, max error {err:.1e} of range")
