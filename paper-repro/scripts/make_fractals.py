"""Generate synthetic fractal images (random iterated function systems) as an IPMix mixing set.

IPMix's authors say any unlabeled synthetic images work as the mixing set; this replaces their
Google Drive set, which cannot be downloaded automatically. Usage:
    python scripts/make_fractals.py <out_dir> [count]
Writes <out_dir>/fractals/synthetic/NNNN.png: pass <out_dir>/fractals as the mixing set (torchvision ImageFolder needs the images in a class subfolder).
"""
import os
import sys

import numpy as np
from PIL import Image


def fractal(rng, size=256, points=200_000):
    n_maps = rng.integers(2, 5)
    maps = []
    for _ in range(n_maps):
        a = rng.uniform(-1, 1, (2, 2))
        s = np.linalg.svd(a, compute_uv=False)[0]
        a = a / max(s, 1e-6) * rng.uniform(0.3, 0.8)      # keep each map contractive
        maps.append((a, rng.uniform(-1, 1, 2)))
    colors = rng.uniform(0.2, 1.0, (n_maps, 3))
    x, pts, idx = np.zeros(2), np.empty((points, 2)), np.empty(points, dtype=int)
    choice = rng.integers(0, n_maps, points)
    for i, k in enumerate(choice):
        a, b = maps[k]
        x = a @ x + b
        pts[i], idx[i] = x, k
    pts, idx = pts[100:], idx[100:]
    lo, hi = pts.min(0), pts.max(0)
    xy = ((pts - lo) / np.maximum(hi - lo, 1e-9) * (size - 1)).astype(int)
    img = np.zeros((size, size, 3))
    np.add.at(img, (xy[:, 1], xy[:, 0]), colors[idx])
    img = np.log1p(img)                                   # density -> brightness, so sparse areas still show
    img = np.clip(img / max(np.percentile(img[img > 0], 99), 1e-9), 0, 1) ** 0.6
    pil = Image.fromarray((img * 255).astype(np.uint8))
    return pil.resize((size // 2, size // 2), Image.LANCZOS).resize((size, size), Image.LANCZOS)  # thicken thin strokes


if __name__ == "__main__":
    out, count = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    d = os.path.join(out, "fractals", "synthetic")
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(count):
        fractal(rng).save(os.path.join(d, f"{i:04d}.png"))
    print(f"wrote {count} images to {d}")
