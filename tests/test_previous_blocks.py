import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from nirs_bci import experiments


class PreviousBlockWindows(unittest.TestCase):
    def test_windows_and_session_history(self):
        # Each block has two three-sample trials; a two-sample window leaves one tail.
        trials = pd.DataFrame({
            "trial_id": range(4), "block_id": [0, 0, 1, 1],
            "label": [2, 3, 2, 3], "start_idx": [0, 3, 6, 9], "end_idx": [3, 6, 9, 12],
        })
        sessions = [
            ({"file_name": str(i), "sampling_rate": 2, "trials": trials},
             (np.arange(12) + offset)[:, None])
            for i, offset in enumerate([0, 100])
        ]
        with patch.object(experiments, "fit_and_predict", return_value=(1, 1)) as fit:
            exp1, exp2 = experiments.run_one_permutation(
                [1, 0], "left_vs_right", "LDA", "none", {"none": sessions}
            )

        # Calls follow exp1, exp2 order, skipping variants with no training block.
        histories = [[100], [100], [100, 106], [0], [100, 106, 0]]
        self.assertEqual(exp1.status.tolist(), ["no_train_random", "ok"] * 2)
        self.assertEqual(exp2.status.tolist(), ["no_train_random", "ok", "ok", "ok"])
        self.assertEqual(fit.call_count, len(histories))
        for call, history in zip(fit.call_args_list, histories):
            inputs = call.kwargs
            starts = [block + delta for block in history for delta in (0, 3)]
            expected = [[start, start + 1] for start in starts]
            np.testing.assert_array_equal(np.asarray(inputs["train_windows"])[:, :, 0], expected)
            np.testing.assert_array_equal(inputs["y_train_windows"], [0, 1] * len(history))
            np.testing.assert_array_equal(inputs["X_train"][:, 0],
                                          np.concatenate([np.arange(b, b + 6) for b in history]))
