from pathlib import Path
import runpy

import numpy as np


def test_train_xor_learns_the_dataset():
    module = runpy.run_path(str(Path(__file__).resolve().parents[1] / "example-nn.py"))

    result = module["train_xor"](epochs=1500, lr=0.1, seed=0, hidden_size=8)

    assert result["accuracy"] == 1.0
    assert result["losses"][0] > result["losses"][-1]
    np.testing.assert_array_equal(result["predictions"], np.array([0, 1, 1, 0]))
