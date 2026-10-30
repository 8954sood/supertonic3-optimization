# Third-party notices

Original optimization, evaluation and training code is MIT licensed; see [LICENSE](LICENSE).

- Supertonic Python components: Supertone Inc., MIT. Runtime imports the installed `supertonic` package. See [code license](docs/licenses/SUPERTONIC_CODE_LICENSE).
- Supertonic 3 and derivative model weights: BigScience Open RAIL-M, independent of this repository's MIT license. See [model license](docs/licenses/SUPERTONIC_MODEL_LICENSE) and [upstream model](https://huggingface.co/Supertone/supertonic-3). Weights are downloaded or trained separately.
- ONNX Runtime C API headers under `native/include/`: Microsoft Corporation, MIT. See [license](native/include/LICENSE.onnxruntime). The native integer and row-parallel normalization kernels are repository code; their supported platforms and validation are recorded separately.
- Mesmer and Paradee are research references. Their source bundles, distributed binaries and model weights are not vendored here.

Dependency packages retain their respective licenses. Synthetic training assets, audio and checkpoints are excluded from Git.
