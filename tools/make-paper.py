"""Generate a seamless tile of plain, slightly warm writing paper.

Usage: python3 make-paper.py [seed] [out.jpg]
Layers: soft mottling (uneven pulp), fine grain, and a few faint fibres.
Every layer wraps around the tile edges, so the tile repeats without seams.
"""
import sys
import numpy as np
from PIL import Image

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OUT = sys.argv[2] if len(sys.argv) > 2 else "paper.jpg"
N = 1024                       # shown at 512 CSS px, so it stays sharp on high-density screens
BASE = np.array([251, 248, 241], float)   # #fbf8f1
rng = np.random.default_rng(SEED)


def periodic_noise(n, lo, hi):
    """White noise band-passed in the Fourier domain: periodic by construction."""
    f = np.fft.fftfreq(n)
    fx, fy = np.meshgrid(f, f)
    r = np.hypot(fx, fy) * n
    band = np.exp(-(r / hi) ** 2) * (1 - np.exp(-(r / lo) ** 2)) if lo else np.exp(-(r / hi) ** 2)
    spec = np.fft.fft2(rng.standard_normal((n, n))) * band
    out = np.real(np.fft.ifft2(spec))
    return out / (out.std() + 1e-9)


light = np.zeros((N, N))
light += 1.6 * periodic_noise(N, 0, 6)       # broad, soft unevenness
light += 0.9 * periodic_noise(N, 6, 40)      # pulp clouds
light += 1.2 * periodic_noise(N, 120, 600)   # fine grain

# faint fibres: short, gently curved strokes, a little darker or lighter than the sheet
fib = np.zeros((N, N))
for _ in range(260):
    x, y = rng.uniform(0, N, 2)
    a = rng.uniform(0, np.pi)
    length = rng.uniform(14, 60)
    curve = rng.uniform(-0.02, 0.02)
    tone = rng.choice([-1, 1], p=[0.65, 0.35]) * rng.uniform(1.2, 2.6)
    steps = int(length * 2)
    for i in range(steps):
        a += curve
        x += np.cos(a) * 0.5
        y += np.sin(a) * 0.5
        fib[int(y) % N, int(x) % N] = tone
# soften the fibres slightly (periodic blur via FFT)
f = np.fft.fftfreq(N)
fx, fy = np.meshgrid(f, f)
fib = np.real(np.fft.ifft2(np.fft.fft2(fib) * np.exp(-(np.hypot(fx, fy) * N / 260) ** 2)))

shade = light + fib * 1.4
img = BASE[None, None, :] + shade[..., None] * np.array([1.0, 1.0, 1.15])[None, None, :]
img = np.clip(img, 0, 255).astype(np.uint8)
Image.fromarray(img, "RGB").save(OUT, quality=82, optimize=True, progressive=False)
print(OUT, img.mean(axis=(0, 1)).round(1), img.min(), img.max())
