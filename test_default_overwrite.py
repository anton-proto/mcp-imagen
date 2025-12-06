#!/usr/bin/env python3
"""Test script to verify default overwrite behavior is now True."""

from pathlib import Path
from src.mcp_imagen_server.imagen_client import ImagenClient

test_image = "/home/d057039/d057039-github/snippets/review-2025/image-prompts-2/02_SL_Analytics_Assistant/nobg_A simple 3D robot or chat bot character with a fri_1.png"
output_dir = "/home/d057039/anton-proto/mcp-imagen/test_default_output"
Path(output_dir).mkdir(exist_ok=True)
output_file = f"{output_dir}/test_default.png"

print("Testing default overwrite behavior (should be True)...")
print()

# Test 1: Create initial file
print("Test 1: Create initial file")
result = ImagenClient.autocrop_image(test_image, output_path=output_file)
print(f"✓ Created: {result}")
print()

# Test 2: Run again WITHOUT specifying overwrite - should succeed (default=True)
print("Test 2: Run again without specifying overwrite (should overwrite)")
try:
    result = ImagenClient.autocrop_image(test_image, output_path=output_file, padding=5)
    print(f"✓ Success - overwrote by default: {result}")
except FileExistsError as e:
    print(f"✗ Failed - should have overwritten by default!")
    print(f"  Error: {e}")
print()

# Test 3: Explicitly set overwrite=False - should fail
print("Test 3: Explicitly set overwrite=False (should fail)")
try:
    result = ImagenClient.autocrop_image(test_image, output_path=output_file, overwrite=False)
    print(f"✗ Unexpected success - should have failed!")
except FileExistsError as e:
    print(f"✓ Correctly prevented overwrite: {e}")
print()

print("All tests passed! Default overwrite is True.")
