"""Preparation must preserve source files, including when hard links are used."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import onnx
from onnx import TensorProto, helper

from prepare_condition_cache import prepare


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        (self.source / "onnx").mkdir(parents=True)
        graph = helper.make_graph([
            helper.make_node("Mul", ["text", "scale"], ["condition"]),
            helper.make_node("Add", ["noisy_latent", "condition"], ["out"]),
        ], "test", [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2])
                    for name in ["text", "noisy_latent"]],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor("scale", TensorProto.FLOAT, [], [2.0])])
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
                  self.source / "onnx/vector_estimator.onnx")
        (self.source / "manifest.json").write_text(json.dumps({"supported_steps": [8], "status": "test"}))
        (self.source / "onnx/transformation.json").write_text('{"prior": true}')
        (self.source / "untouched.txt").write_text("immutable asset")
        self.before = {path.relative_to(self.source): path.read_bytes()
                       for path in self.source.rglob("*") if path.is_file()}

    def assert_source_unchanged(self):
        self.assertEqual(self.before, {path.relative_to(self.source): path.read_bytes()
                                      for path in self.source.rglob("*") if path.is_file()})

    def test_copy_and_hard_link_preserve_source_and_metadata(self):
        for linked in [False, True]:
            output = self.root / ("linked" if linked else "copied")
            receipt = prepare(self.source, output, link_assets=linked)
            self.assertEqual(receipt["initializer_count"], 1)
            self.assertTrue(receipt["initializer_bitidentity"])
            self.assertTrue((output / "onnx/condition.onnx").exists())
            metadata = json.loads((output / "manifest.json").read_text())
            self.assertEqual(metadata["status"], "test")
            self.assertEqual(metadata["supported_steps"], [8])
            self.assertEqual(metadata["vector_sha256"], hashlib.sha256(
                (output / "onnx/vector_estimator.onnx").read_bytes()).hexdigest())
            self.assertEqual((output / "untouched.txt").stat().st_ino ==
                             (self.source / "untouched.txt").stat().st_ino, linked)
            self.assert_source_unchanged()

    def test_failed_split_is_atomic_and_does_not_touch_source(self):
        output = self.root / "failed"
        with patch("prepare_condition_cache.split", side_effect=RuntimeError("Failed split")):
            with self.assertRaisesRegex(RuntimeError, "Failed split"):
                prepare(self.source, output, link_assets=True)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".condition-cache-*")))
        self.assert_source_unchanged()

    def test_reject_existing_nested_and_already_cached_packages(self):
        for output in [self.source, self.source / "nested", self.root]:
            with self.assertRaises(ValueError):
                prepare(self.source, output)
        output = self.root / "cached"
        prepare(self.source, output)
        with self.assertRaisesRegex(ValueError, "already has"):
            prepare(output, self.root / "double")
        self.assert_source_unchanged()


if __name__ == "__main__":
    unittest.main()
