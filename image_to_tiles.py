import os
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image

input_tif = "Agartala.tif"
output_folder = "tiles_output_bright"
tile_width, tile_height = 512, 512

os.makedirs(output_folder, exist_ok=True)

def scale_to_uint8_percentile(tile, p2, p98):
    """Convert tile to uint8 using 2nd and 98th percentiles per band."""
    out = np.empty_like(tile, dtype=np.uint8)
    for i in range(3):
        band = tile[i]
        band = np.clip(band, p2[i], p98[i])
        out[i] = ((band - p2[i]) / (p98[i] - p2[i]) * 255).astype(np.uint8)
    return out

with rasterio.open(input_tif) as src:
    band_indexes = [1, 2, 3]

    # Sample a chunk of the image to estimate percentiles
    print("📊 Sampling image to compute 2nd and 98th percentiles...")
    sample = src.read(band_indexes, out_shape=(3, 512, 512))
    p2 = np.percentile(sample, 2, axis=(1, 2))
    p98 = np.percentile(sample, 98, axis=(1, 2))
    print(f"✅ Percentiles (2%–98%) per band: {list(zip(p2, p98))}")

    tile_count = 0
    for top in range(0, src.height, tile_height):
        for left in range(0, src.width, tile_width):
            width = min(tile_width, src.width - left)
            height = min(tile_height, src.height - top)
            window = Window(left, top, width, height)
            tile = src.read(band_indexes, window=window)

            tile_uint8 = scale_to_uint8_percentile(tile, p2, p98)
            tile_image = tile_uint8.transpose((1, 2, 0))  # (C, H, W) -> (H, W, C)

            tile_path = os.path.join(output_folder, f"tile_{tile_count:04d}.png")
            Image.fromarray(tile_image).save(tile_path)
            tile_count += 1

print(f"✅ Done! Saved {tile_count} tiles with enhanced brightness using percentile stretching.")
