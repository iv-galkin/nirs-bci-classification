"""Lightweight interface and model configuration checks; no real datasets."""

import importlib
import inspect
import unittest

from nirs_bci import experiments, models


MODULES = (
    "boosting", "channel_selection", "data", "evaluation", "experiments",
    "features", "models", "nitrc", "preprocessing", "segmentation",
    "spectral", "visualization",
)


class PackageInterfaces(unittest.TestCase):
    def test_all_modules_import(self):
        for name in MODULES:
            with self.subTest(module=name):
                importlib.import_module(f"nirs_bci.{name}")

    def test_factory_lda_qda_and_unknown_name(self):
        for mode in ("LDA", "QDA"):
            self.assertEqual(models.make_classifier(mode).mode, mode)
        with self.assertRaises(ValueError):
            models.make_classifier("Unknown")

    def test_catboost_variant_parameters(self):
        expected = {
            "CatBoost_mean": (200, 3, 0.05, 10),
            "CatBoost_mean_std": (200, 3, 0.05, 10),
            "CatBoost_basic": (200, 3, 0.05, 10),
            "CatBoostTiny_mean": (30, 1, 0.03, 100),
            "CatBoostConservative_mean": (200, 2, 0.03, 30),
        }
        keys = ("iterations", "depth", "learning_rate", "l2_leaf_reg")
        for name, values in expected.items():
            with self.subTest(classifier=name):
                params = models.make_classifier(name).get_params()
                self.assertEqual(tuple(params[key] for key in keys), values)
                self.assertEqual(params["loss_function"], "Logloss")
                self.assertEqual(params["random_seed"], 42)
        params = models.make_catboost_tiny_model(random_seed=7).get_params()
        self.assertEqual(params["random_seed"], 7)
        self.assertEqual(params["min_data_in_leaf"], 10)

    def test_svm_variant_parameters(self):
        for name, kernel in (("SVM_linear_mean", "linear"), ("SVM_rbf_mean_std", "rbf")):
            with self.subTest(classifier=name):
                steps = models.make_classifier(name).named_steps
                self.assertEqual(list(steps), ["scaler", "svm"])
                params = steps["svm"].get_params()
                self.assertEqual(
                    (params["kernel"], params["C"], params["class_weight"], params["random_state"]),
                    (kernel, 1.0, "balanced", 42),
                )
                if kernel == "rbf":
                    self.assertEqual(params["gamma"], "scale")

    def test_feature_configuration_remains_explicit(self):
        self.assertIsNone(
            inspect.signature(experiments.fit_and_predict).parameters["feature_set"].default
        )
        self.assertEqual(experiments.FEATURE_SET, {})


if __name__ == "__main__":
    unittest.main()
