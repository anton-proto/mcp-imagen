#!/usr/bin/env python3
"""Test script for autocrop functionality."""

import sys
from pathlib import Path

from src.mcp_imagen_server.imagen_client import ImagenClient

# Test images
test_images = [
    "/home/d057039/d057039-github/snippets/review-2025/image-prompts-2/02_SL_Analytics_Assistant/nobg_A simple 3D robot or chat bot character with a fri_1.png",
    "/home/d057039/d057039-github/snippets/review-2025/image-prompts-2/02_SL_Analytics_Assistant/nobg_A simple 3D robot or chat bot character with a fri_2.png",
    "/home/d057039/d057039-github/snippets/review-2025/image-prompts-2/02_SL_Analytics_Assistant/nobg_A simple 3D robot or chat bot character with a fri_3.png",
]

# Output directory
output_dir = "/home/d057039/anton-proto/mcp-imagen-autocrop/test_output"
Path(output_dir).mkdir(exist_ok=True)

print("Testing autocrop functionality...")
print(f"Processing {len(test_images)} images")
print()

# Test 1: Single image without padding
print("Test 1: Single image without padding")
try:
    result = ImagenClient.autocrop_image(
        test_images[0],
        output_path=f"{output_dir}/test1_cropped.png",
        padding=0,
    )
    print(f"✓ Success: {result}")
except Exception as e:
    print(f"✗ Failed: {e}")
print()

# Test 2: Single image with 10px padding
print("Test 2: Single image with 10px padding")
try:
    result = ImagenClient.autocrop_image(
        test_images[1],
        output_path=f"{output_dir}/test2_cropped_padding.png",
        padding=10,
    )
    print(f"✓ Success: {result}")
except Exception as e:
    print(f"✗ Failed: {e}")
print()

# Test 3: Default output path (same directory)
print("Test 3: Default output path (same directory as input)")
try:
    result = ImagenClient.autocrop_image(
        test_images[2],
        padding=5,
    )
    print(f"✓ Success: {result}")
except Exception as e:
    print(f"✗ Failed: {e}")
print()

print("All tests completed!")
print(f"Check output directory: {output_dir}")
