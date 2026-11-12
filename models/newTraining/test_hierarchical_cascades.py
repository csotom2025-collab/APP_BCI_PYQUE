import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from models.newTraining.BCIpredictorhierarchical import HierarchicalBCIPredictor
from models.newTraining.train_hierarchical_optimized import (
    build_top_hierarchical_bundles,
    _prepare_color_training_data,
)


def _fixed_bundle(label):
    return {
        "fixed_label": label,
        "feature_columns": [],
        "feature_set": "N/A",
    }


def _top_models():
    models = {"_cascada_color": [_fixed_bundle("Green") for _ in range(5)]}
    for group, color in (
        ("Letters", "Green"),
        ("Numbers", "Blue"),
        ("Controls", "Blue"),
    ):
        models[f"_cascada_color_{group}"] = [
            _fixed_bundle(color) for _ in range(5)
        ]
    for color in ("Green", "Blue"):
        models[f"_cascada_super_{color}"] = [
            _fixed_bundle("Letters") for _ in range(5)
        ]
        for group, label in (
            ("Letters", "A"),
            ("Numbers", "1"),
            ("Controls", "Enter"),
        ):
            models[f"_cascada_fino_{color}_{group}"] = [
                _fixed_bundle(label) for _ in range(5)
            ]
    return models


class HierarchicalCascadeTests(unittest.TestCase):
    def test_color_training_data_is_limited_to_requested_group(self):
        samples = pd.DataFrame({
            "tpComando": ["Letters", "Letters", "Numbers", "Numbers"],
            "color": ["Green", "Blue", "Green", "Blue"],
            "feature": [1.0, 2.0, 3.0, 4.0],
        })

        letters = _prepare_color_training_data(samples, group="Letters")
        self.assertEqual(set(letters["tpComando"]), {"Letters"})
        self.assertEqual(letters["label"].tolist(), ["Green", "Blue"])

        complete = _prepare_color_training_data(samples)
        self.assertEqual(len(complete), len(samples))

        with self.assertRaisesRegex(ValueError, "requiere muestras Green y Blue"):
            _prepare_color_training_data(
                samples[samples["tpComando"] == "Letters"].iloc[:1],
                group="Letters",
            )

    def test_builds_five_bundles_for_each_cascade_type(self):
        bundles = build_top_hierarchical_bundles(
            _top_models(), {"channel_names": ["Cz"], "fs": 128}
        )

        self.assertEqual(
            set(bundles), {"completas", "letras", "numeros", "comandos"}
        )
        self.assertTrue(all(len(items) == 5 for items in bundles.values()))
        self.assertIn(
            "super_clase",
            bundles["completas"][0]["por_color"]["Green"],
        )
        for category, group in (
            ("letras", "Letters"),
            ("numeros", "Numbers"),
            ("comandos", "Controls"),
        ):
            hierarchy = bundles[category][0]["por_color"]["Green"]
            self.assertNotIn("super_clase", hierarchy)
            self.assertEqual(list(hierarchy["por_grupo"]), [group])

    def test_all_cascade_types_load_and_predict(self):
        bundles = build_top_hierarchical_bundles(
            _top_models(), {"channel_names": ["Cz"], "fs": 128}
        )
        expected = {
            "completas": ("Letters", "A", "Green"),
            "letras": ("Letters", "A", "Green"),
            "numeros": ("Numbers", "1", "Blue"),
            "comandos": ("Controls", "Enter", "Blue"),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            for cascade_type, (group, command, color) in expected.items():
                bundle = bundles[cascade_type][0]
                bundle["color"]["f1_macro_cv"] = 0.0
                for hierarchy in bundle["por_color"].values():
                    if "super_clase" in hierarchy:
                        hierarchy["super_clase"]["f1_macro_cv"] = 0.0
                if cascade_type == "completas":
                    bundle.pop("cascade_type")

                model_path = Path(temp_dir) / f"{cascade_type}.joblib"
                joblib.dump(bundle, model_path)
                with contextlib.redirect_stdout(io.StringIO()):
                    predictor = HierarchicalBCIPredictor(
                        model_path=str(model_path)
                    )

                predictor._signals_to_features = (
                    lambda signals, return_artifact_info=False: (
                        pd.DataFrame({"x": [1.0, 2.0]}),
                        False,
                        0.0,
                    )
                )
                predictor.last_quality_filter_info = {}
                prediction, details = predictor.predict_from_repetitions(
                    [object()]
                )

                self.assertEqual(prediction, command)
                expected_type = (
                    "completa" if cascade_type == "completas" else cascade_type
                )
                self.assertEqual(details["cascade_type"], expected_type)
                self.assertEqual(details["grupo_predicho"], group)
                self.assertEqual(details["color_predicho"], color)


if __name__ == "__main__":
    unittest.main()
