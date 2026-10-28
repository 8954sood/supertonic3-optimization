"""Differentiable interpreter for the original Supertonic vector ONNX graph.
Experimental training bridge, not a replacement inference engine. Parity required.
"""

import torch
from torch import nn
from torch.nn import functional as F
import onnx
from onnx import numpy_helper, helper
import numpy as np


def integers(value):
    return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]


class VectorGraph(nn.Module):
    def __init__(
        self,
        path,
        device="mps",
        checkpoints=None,
        pair_manifest="distill/data/manifest.json",
    ):
        super().__init__()
        graph = onnx.load(path).graph
        self.nodes = [
            (
                n.name,
                n.op_type,
                list(n.input),
                list(n.output),
                {a.name: helper.get_attribute_value(a) for a in n.attribute},
            )
            for n in graph.node
        ]
        self.outputs = [v.name for v in graph.output]
        self.names = {}
        self.trainable = nn.ParameterDict()
        replacements = {}
        if checkpoints and __import__("pathlib").Path(checkpoints).is_file():
            replacements = torch.load(
                checkpoints, map_location="cpu", weights_only=True
            )["weights"]
        elif checkpoints:
            import json
            from pathlib import Path

            pairs = json.loads(Path(pair_manifest).read_text())["pairs"]
            for i, pair in enumerate(pairs):
                saved = torch.load(
                    Path(checkpoints) / f"layer_{i:02d}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                state = saved["float_state"]
                for name, value in [
                    (pair["w1"], state["fc1.weight"][:, :, None]),
                    (pair["b1"], state["fc1.bias"]),
                    (pair["w2"], state["fc2.weight"][:, :, None]),
                    (pair["b2"], state["fc2.bias"]),
                ]:
                    replacements[name] = value
        for i, init in enumerate(graph.initializer):
            value = replacements.get(init.name)
            if value is None:
                value = torch.from_numpy(
                    np.array(numpy_helper.to_array(init), copy=True)
                )
            key = f"w{i}"
            if init.name in replacements:
                self.trainable[key] = nn.Parameter(value.to(device))
            else:
                self.register_buffer(
                    key, value.to(device) if value.is_floating_point() else value
                )
            self.names[init.name] = key
        self.device_name = device
        self.qat = False

    def forward(self, feed):
        env = {
            name: self.trainable[key] if key in self.trainable else getattr(self, key)
            for name, key in self.names.items()
        }
        for name, value in feed.items():
            if not isinstance(value, torch.Tensor):
                value = torch.from_numpy(value)
            env[name] = value.to(self.device_name)
        for name, op, inputs, outputs, a in self.nodes:
            x = [env[k] if k else None for k in inputs]
            try:
                result = self.operation(op, x, a, len(outputs))
            except Exception as e:
                raise RuntimeError(f"{name} ({op}): {e}") from e
            if not isinstance(result, tuple):
                result = (result,)
            for key, value in zip(outputs, result):
                env[key] = value
        return [env[k] for k in self.outputs]

    def operation(self, op, x, a, nout):
        if op == "Identity":
            return x[0]
        if op in ("Add", "Sub", "Mul", "Div", "Equal", "MatMul"):
            u, v = x[:2]
            if u.device != v.device:
                device = u.device if u.device.type != "cpu" else v.device
                u, v = u.to(device), v.to(device)
            if op == "Add":
                return u + v
            if op == "Sub":
                return u - v
            if op == "Mul":
                return u * v
            if op == "Equal":
                return u == v
            if op == "MatMul":
                return u @ v
            return (
                torch.div(u, v, rounding_mode="trunc")
                if not u.is_floating_point() and not v.is_floating_point()
                else u / v
            )
        if op == "Transpose":
            return x[0].permute(a["perm"])
        if op == "Shape":
            return torch.tensor(
                list(x[0].shape)[a.get("start", 0) : a.get("end", x[0].ndim)],
                dtype=torch.int64,
            )
        if op == "Reshape":
            shape = integers(x[1])
            shape = [
                x[0].shape[i] if v == 0 and not a.get("allowzero", 0) else v
                for i, v in enumerate(shape)
            ]
            return x[0].reshape(shape)
        if op == "Slice":
            start, end = integers(x[1]), integers(x[2])
            axes = integers(x[3]) if len(x) > 3 else list(range(len(start)))
            steps = integers(x[4]) if len(x) > 4 else [1] * len(start)
            slices = [slice(None)] * x[0].ndim
            value = x[0]
            for b, e, axis, step in zip(start, end, axes, steps):
                first, last, stride = slice(b, e, step).indices(value.shape[axis])
                if stride < 0:
                    index = torch.tensor(
                        list(range(first, last, stride)), device=value.device
                    )
                    value = torch.index_select(value, axis, index)
                else:
                    slices[axis] = slice(first, last, stride)
            return value[tuple(slices)]
        if op == "Gather":
            axis = a.get("axis", 0)
            idx = x[1].long().to(x[0].device)
            selected = torch.index_select(x[0], axis, idx.reshape(-1))
            return selected.reshape(
                tuple(x[0].shape[:axis])
                + tuple(idx.shape)
                + tuple(x[0].shape[axis + 1 :])
            )
        if op == "Concat":
            device = next((v.device for v in x if v.device.type != "cpu"), x[0].device)
            return torch.cat([v.to(device) for v in x], dim=a["axis"])
        if op == "Unsqueeze":
            y = x[0]
            axes = integers(x[1]) if len(x) > 1 else a["axes"]
            ndim = y.ndim + len(axes)
            for axis in sorted(v % ndim for v in axes):
                y = y.unsqueeze(axis)
            return y
        if op == "Squeeze":
            y = x[0]
            axes = integers(x[1]) if len(x) > 1 else a.get("axes")
            if axes is None:
                return y.squeeze()
            for axis in sorted([v % y.ndim for v in axes], reverse=True):
                y = y.squeeze(axis)
            return y
        if op == "Tile":
            return x[0].repeat(integers(x[1]))
        if op == "Expand":
            shape = integers(x[1])
            orig = [1] * (len(shape) - x[0].ndim) + list(x[0].shape)
            return x[0].expand([max(v, w) for v, w in zip(shape, orig)])
        if op == "Where":
            device = next((v.device for v in x if v.device.type != "cpu"), x[0].device)
            return torch.where(*[v.to(device) for v in x])
        if op == "Cast":
            dtype = {1: torch.float32, 7: torch.int64, 9: torch.bool, 6: torch.int32}[
                a["to"]
            ]
            return x[0].to(dtype=dtype)
        if op == "ConstantOfShape":
            value = (
                numpy_helper.to_array(a["value"])
                if "value" in a
                else np.array([0], np.float32)
            )
            return torch.full(
                integers(x[0]), float(value.item()), device=self.device_name
            )
        if op == "Conv":
            pads = a.get("pads", [0, 0])
            u = x[0]
            weight = x[1]
            if self.qat and weight.shape[-1] == 1 and a.get("group", 1) == 1:
                from distill.train import fake_activation, fake_weight

                u = fake_activation(u)
                weight = fake_weight(weight[:, :, 0])[:, :, None]
            if pads[0] != pads[1]:
                u = F.pad(u, pads)
                pads = [0, 0]
            return F.conv1d(
                u,
                weight,
                x[2] if len(x) > 2 else None,
                stride=a.get("strides", [1])[0],
                padding=pads[0],
                dilation=a.get("dilations", [1])[0],
                groups=a.get("group", 1),
            )
        if op == "Pad":
            pads = integers(x[1])
            rank = len(pads) // 2
            p = []
            for i in reversed(range(rank)):
                p.extend([pads[i], pads[i + rank]])
            while len(p) > 2 and p[-2:] == [0, 0]:
                p = p[:-2]
            mode = a.get("mode", b"constant").decode()
            mode = {"edge": "replicate"}.get(mode, mode)
            return (
                F.pad(x[0], p, mode=mode, value=float(x[2]) if len(x) > 2 else 0)
                if mode == "constant"
                else F.pad(x[0], p, mode=mode)
            )
        if op == "LayerNormalization":
            axis = a.get("axis", -1) % x[0].ndim
            return F.layer_norm(
                x[0],
                x[0].shape[axis:],
                x[1],
                x[2] if len(x) > 2 else None,
                a.get("epsilon", 1e-5),
            )
        if op == "Gemm":
            u = x[0].T if a.get("transA", 0) else x[0]
            v = x[1].T if a.get("transB", 0) else x[1]
            return a.get("alpha", 1) * (u @ v) + (
                a.get("beta", 1) * x[2] if len(x) > 2 else 0
            )
        if op == "Split":
            axis = a.get("axis", 0)
            sizes = integers(x[1]) if len(x) > 1 else a.get("split")
            return (
                tuple(torch.split(x[0], sizes, dim=axis))
                if sizes
                else tuple(torch.chunk(x[0], nout, dim=axis))
            )
        if op == "ReduceSum":
            axes = (
                tuple(integers(x[1]))
                if len(x) > 1
                else tuple(a.get("axes", range(x[0].ndim)))
            )
            return x[0].sum(dim=axes, keepdim=bool(a.get("keepdims", 1)))
        if op == "Softmax":
            return F.softmax(x[0], dim=a.get("axis", -1))
        if op == "Constant":
            value = torch.from_numpy(
                np.array(numpy_helper.to_array(a["value"]), copy=True)
            )
            return value.to(self.device_name) if value.is_floating_point() else value
        if op == "BatchNormalization":
            return F.batch_norm(
                x[0], x[3], x[4], x[1], x[2], training=False, eps=a.get("epsilon", 1e-5)
            )
        if op == "PRelu":
            return F.prelu(x[0], x[1].reshape(-1))
        functions = {
            "Erf": torch.erf,
            "Tanh": torch.tanh,
            "Sin": torch.sin,
            "Cos": torch.cos,
            "Reciprocal": torch.reciprocal,
            "Softplus": F.softplus,
        }
        if op in functions:
            return functions[op](x[0])
        raise NotImplementedError(op)

    def export_weights(self):
        return {
            name: self.trainable[key].detach().cpu()
            for name, key in self.names.items()
            if key in self.trainable
        }

    def train_full_vector(self):
        """Unfreeze learned vector-field tensors, preserving shape constants."""
        for name, key in self.names.items():
            if (
                name.startswith("vector_estimator.tts.ttl.vector_field.")
                and name.endswith((".weight", ".bias", ".gamma"))
                and key not in self.trainable
            ):
                value = self._buffers.pop(key)
                self.trainable[key] = nn.Parameter(value)

    def train_full_decoder(self):
        for name, key in self.names.items():
            if (
                name.startswith("tts.ae.decoder.")
                and name.endswith((".weight", ".bias", ".gamma"))
                and key not in self.trainable
            ):
                value = self._buffers.pop(key)
                self.trainable[key] = nn.Parameter(value)

    def load_weights(self, weights):
        with torch.no_grad():
            for name, value in weights.items():
                key = self.names[name]
                target = (
                    self.trainable[key] if key in self.trainable else getattr(self, key)
                )
                target.copy_(value.to(target.device))
