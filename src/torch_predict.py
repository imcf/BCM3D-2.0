"""PyTorch copy of the BCM3D 2.0 CNN, for GPU prediction without TensorFlow.

TensorFlow >= 2.11 has no GPU support on native Windows; PyTorch does. The
network (CSBDeep CARE residual 3D U-Net, depth 2, 32 first filters) is
rebuilt in PyTorch and loads the original Keras ``weights_best.h5``, so
predictions match ``CARE.predict`` (identical in float32; within ~0.1 % of
the value range with PyTorch's default TF32 convolutions on Ampere+ GPUs).

Example::

    from src.torch_predict import load_model, predict
    net = load_model("models/bcm3d_ecoli")        # cuda if available
    pred = predict(net, vol)                       # ZCYX, C = 2 maps
"""
import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

_CONVS = [  # (keras layer name, in, out) of the CSBDeep 3D U-Net
    ("down_level_0_no_0", 2, 32), ("down_level_0_no_1", 32, 32),
    ("down_level_1_no_0", 32, 64), ("down_level_1_no_1", 64, 64),
    ("middle_0", 64, 128), ("middle_2", 128, 64),
    ("up_level_1_no_0", 128, 64), ("up_level_1_no_2", 64, 32),
    ("up_level_0_no_0", 64, 32), ("up_level_0_no_2", 32, 32),
]


class Net(nn.Module):
    """Residual 3D U-Net with the same layout as the BCM3D CARE models."""

    def __init__(self):
        super().__init__()
        self.c = nn.ModuleDict({n: nn.Conv3d(i, o, 3, padding=1)
                                for n, i, o in _CONVS})
        self.last = nn.Conv3d(32, 2, 1)

    def _two(self, h, a, b):
        return F.relu(self.c[b](F.relu(self.c[a](h))))

    def forward(self, x):
        """Map a (N, 2, Z, Y, X) tensor to the two output channels."""
        s0 = self._two(x, "down_level_0_no_0", "down_level_0_no_1")
        s1 = self._two(F.max_pool3d(s0, 2), "down_level_1_no_0",
                       "down_level_1_no_1")
        h = self._two(F.max_pool3d(s1, 2), "middle_0", "middle_2")
        h = torch.cat([F.interpolate(h, scale_factor=2), s1], 1)
        h = self._two(h, "up_level_1_no_0", "up_level_1_no_2")
        h = torch.cat([F.interpolate(h, scale_factor=2), s0], 1)
        h = self._two(h, "up_level_0_no_0", "up_level_0_no_2")
        return self.last(h) + x  # residual, linear output


def load_model(model_dir, device=None):
    """Build the network and load the Keras weights of a BCM3D model.

    Parameters:
        model_dir (str): Folder holding ``weights_best.h5`` (OSF m4637).
        device (str): "cuda" or "cpu". None = "cuda" if available.

    Returns:
        Net: The network in eval mode on ``device``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    net = Net()
    layers = [(n, net.c[n]) for n, *_ in _CONVS] + [("conv3d_1", net.last)]
    with h5py.File(f"{model_dir}/weights_best.h5", "r") as f:
        for name, conv in layers:
            g = f[name][name]
            k = torch.from_numpy(g["kernel:0"][()])  # (z, y, x, in, out)
            conv.weight.data = k.permute(4, 3, 0, 1, 2).contiguous()
            conv.bias.data = torch.from_numpy(g["bias:0"][()])
    return net.eval().to(device)


def predict(net, vol, tile=128, margin=32):
    """Predict the two intermediate maps of a ZYX volume.

    Same pre/post steps as CSBDeep ``CARE.predict``: 2-99.8 percentile
    normalisation, reflect padding to a multiple of 4, de-normalisation.
    The same volume is fed to both input channels, as in predict.ipynb.

    Parameters:
        net (Net): Output of :func:`load_model`.
        vol (np.ndarray): ZYX volume (isotropic voxels, as for training).
        tile (int): YX tile size, multiple of 4. GPU memory is about
            1.6 GiB per 50 z slices at 128; 256 is not faster.
        margin (int): Overlap per side, multiple of 4, larger than the
            receptive-field radius (~23 voxels).

    Returns:
        np.ndarray: ZCYX float32 prediction, C = 2.
    """
    mi, ma = np.percentile(vol, [2, 99.8])
    x = ((np.asarray(vol, np.float32) - mi) / (ma - mi + 1e-20)).astype(
        np.float32)
    shape = x.shape
    # pad every axis to a multiple of 4, split like CSBDeep's resizer
    pads = [((4 - s % 4) % 4 // 2, (4 - s % 4) % 4 - (4 - s % 4) % 4 // 2)
            for s in shape]
    x = np.pad(x, pads, mode="reflect")
    yp, xp = x.shape[1:]
    out = np.empty((x.shape[0], 2, yp, xp), np.float32)
    dev = next(net.parameters()).device
    # YX tiles with context clipped at the border: all bounds are multiples
    # of 4, so the pooling grid matches an untiled run
    with torch.inference_mode():
        for y0 in range(0, yp, tile):
            for x0 in range(0, xp, tile):
                ya, xa = max(0, y0 - margin), max(0, x0 - margin)
                yb = min(yp, y0 + tile + margin)
                xb = min(xp, x0 + tile + margin)
                t = torch.from_numpy(np.ascontiguousarray(x[:, ya:yb, xa:xb]))
                t = t[None, None].expand(1, 2, *t.shape).to(dev)
                h, w = min(tile, yp - y0), min(tile, xp - x0)
                p = net(t)[0, :, :, y0 - ya:y0 - ya + h, x0 - xa:x0 - xa + w]
                out[:, :, y0:y0 + h, x0:x0 + w] = \
                    p.permute(1, 0, 2, 3).cpu().numpy()
    (z0, _), (y0, _), (x0, _) = pads
    out = out[z0:z0 + shape[0], :, y0:y0 + shape[1], x0:x0 + shape[2]]
    return out * (ma - mi) + mi
