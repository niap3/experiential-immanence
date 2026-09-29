"""Procedural plank texture for the Experiential Immanence definition strips.

Vertical boards with bevelled seams and staggered butt joints, tileable in both
directions. Lighting (sheen, inner shadows) is left to CSS so the tile stays flat.
Usage: python3 make_wood.py [seed] [out.jpg]
"""
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import zoom, gaussian_filter

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
OUT = sys.argv[2] if len(sys.argv) > 2 else 'wood-gen.jpg'
W, H = 1600, 1000            # 2x pixels for an 800 x 500 CSS tile
rng = np.random.default_rng(SEED)

# muted warm browns (sRGB 0..1): light earlywood, mid, dark latewood/grain, seam
LIGHT = np.array([172, 128, 88]) / 255
MID = np.array([134, 95, 62]) / 255
DARK = np.array([82, 52, 32]) / 255
SEAM = np.array([48, 31, 20]) / 255


def smooth_noise(h, w, cell_y, cell_x):
    """Value noise of shape (h, w), feature size roughly cell_y x cell_x pixels."""
    gh, gw = max(2, int(np.ceil(h / cell_y)) + 3), max(2, int(np.ceil(w / cell_x)) + 3)
    g = rng.standard_normal((gh, gw))
    z = zoom(g, (cell_y, cell_x), order=3)
    oy, ox = rng.integers(0, max(1, z.shape[0] - h)), rng.integers(0, max(1, z.shape[1] - w))
    z = z[oy:oy + h, ox:ox + w]
    return (z - z.mean()) / (z.std() + 1e-6)


def board(w, L):
    """Grain field t (0 light .. 1 dark) and tone multiplier for one board segment w x L."""
    v = np.arange(L)[:, None].astype(float)
    u = np.arange(w)[None, :].astype(float)
    # gentle lateral wander of the grain along the board
    wander = 2.2 * smooth_noise(L, 1, 520, 1) + 0.7 * smooth_noise(L, 1, 140, 1)
    xw = u + wander
    # growth rings: gradual darkening then a sharp edge (earlywood to latewood)
    if rng.random() < 0.45:
        # plain-sawn: nested arches in the middle that straighten toward the edges
        cx = rng.uniform(0.3, 0.7) * w
        cy = rng.uniform(0.0, 1.0) * L
        k = rng.uniform(11, 19)
        f = ((xw - cx) / w) ** 2 * k + (v - cy) / rng.uniform(70, 120) * rng.choice([-1, 1])
        f += 0.35 * smooth_noise(L, w, 200, 40) + 1.1 * smooth_noise(L, 1, 300, 1)
    else:
        # rift or straight grain: long parallel stripes of uneven width
        f = xw / rng.uniform(10, 17) + 0.5 * smooth_noise(L, w, 600, 45)
    r = f % 1.0
    rings = np.clip((r - 0.62) / 0.38, 0, 1) ** 1.7 + 0.25 * r
    # fibres: very fine streaks along the board
    fib = gaussian_filter(rng.standard_normal((L, w)), sigma=(28, 0.55))
    fib = (fib - fib.mean()) / (fib.std() + 1e-6)
    # long soft streaks
    streak = smooth_noise(L, w, 600, 26)
    # pores: sparse, fine, elongated dashes
    n = int(w * L / 2600)
    pores = np.zeros((L, w))
    pores[rng.integers(0, L, n), rng.integers(0, w, n)] = rng.uniform(0.3, 1.0, n)
    pores = gaussian_filter(pores, sigma=(4.5, 0.4))
    t = 0.24 + 0.46 * rings + 0.08 * fib + 0.07 * streak + 3.2 * pores
    tone = 1 + rng.uniform(-0.06, 0.06) + 0.03 * smooth_noise(L, w, 800, 80)
    return np.clip(t, 0, 1.1), tone


img = np.zeros((H, W, 3))
# board widths (px at 2x) that exactly fill W so the tile repeats horizontally
widths = []
while sum(widths) < W:
    widths.append(int(rng.integers(120, 176)))
widths[-1] -= sum(widths) - W
if widths[-1] < 90:                      # fold a sliver into its neighbour
    sliver = widths.pop()
    widths[-1] += sliver
assert sum(widths) == W, widths

x0 = 0
for w in widths:
    # each board slightly warmer or cooler, lighter or darker
    board_tint = np.array([1.0, 1.0, 1.0]) * rng.uniform(0.94, 1.05) + np.array([1, 0, -1]) * rng.uniform(-0.03, 0.03)
    # boards run the full tile; the only joint sits on the tile's top edge, which the
    # torn paper edge covers, so no joint line can pass behind a line of text
    joints = [0]
    starts = list(joints)
    for i, s in enumerate(starts):
        e = starts[(i + 1) % len(starts)]
        L = (e - s) % H or H
        t, tone = board(w, L)
        tt = np.clip(t, 0, 1)[..., None]
        col = np.where(tt < 0.5, LIGHT + (MID - LIGHT) * (tt / 0.5), MID + (DARK - MID) * ((tt - 0.5) / 0.5))
        col = col * board_tint
        col *= tone[..., None]
        # skeuomorphic board shading: slightly rounded face, highlight on the left edge, shade on the right
        uu = np.linspace(-1, 1, w)[None, :]
        col *= (1 - 0.05 * uu ** 2)[..., None]
        col[:, 2:4] = col[:, 2:4] * 1.14
        col[:, -5:] *= np.linspace(0.97, 0.8, 5)[None, :, None]
        # butt joint at the top of each segment: dark line with a highlight below it
        col[0:2] = SEAM
        col[2:3] = col[2:3] * 1.18
        col[3:7] *= np.linspace(0.9, 1.0, 4)[:, None, None]
        rows = (s + np.arange(L)) % H
        img[rows, x0:x0 + w] = col
    # seam between boards: two dark pixels at the board's left edge
    img[:, x0:x0 + 2] = SEAM
    x0 += w

# fine film grain, then clamp and save
img += rng.normal(0, 0.012, img.shape)
img = np.clip(img, 0, 1)
Image.fromarray((img * 255).round().astype(np.uint8)).save(OUT, quality=80, optimize=True, progressive=True)
lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
print(OUT, img.shape, 'mean sRGB', (img.reshape(-1, 3).mean(0) * 255).round(1), 'lum mean', round(float(lum.mean()), 3))
