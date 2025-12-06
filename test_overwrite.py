#!/usr/bin/env python3
"""Test script for autocrop overwrite functionality."""

import sys
from pathlib import Path

from src.mcp_imagen_server.imagen_client import ImagenClient

# Test image
test_image = "/home/d057039/d057039-github/snippets/review-2025/image-prompts-2/02_SL_Analytics_Assistant/nobg_A simple 3D robot or chat bot character with a fri_1.png"

# Output directory
output_dir = "/home/d057039/anton-proto/mcp-imagen-overwrite/test_overwrite_output"
Path(output_dir).mkdir(exist_ok=True)

output_file = f"{output_dir}/test_overwrite.png"

print("Testing autocrop overwrite functionality...")
print()

# Test 1: Create initial file
print("Test 1: Create initial cropped file")
try:
    result = ImagenClient.autocrop_image(
        test_image,
        output_path=output_file,
        padding=0,
        overwrite=False,
    )
    print(f"✓ Success: {result}")
    print(f"  File exists: {Path(output_file).exists()}")
except Exception as e:
    print(f"✗ Failed: {e}")
print()

# Test 2: Try to overwrite without overwrite=True (should fail)
print("Test 2: Try to overwrite without overwrite=True (should fail)")
try:
    result = ImagenClient.autocrop_image(
        test_image,
        output_path=output_file,
        padding=0,
        overwrite=False,
    )
    print(f"✗ Unexpected success - should have failed!")
except FileExistsError as e:
    print(f"✓ Correctly prevented overwrite: {e}")
except Exception as e:
    print(f"✗ Wrong error type: {e}")
print()

# Test 3: Overwrite with overwrite=True (should succeed)
print("Test 3: Overwrite with overwrite=True (should succeed)")
try:
    result = ImagenClient.autocrop_image(
        test_image,
        output_path=output_file,
        padding=5,  # Different padding to verify it's actually re-processing
        overwrite=True,
    )
    print(f"✓ Success: {result}")
    print(f"  File exists: {Path(output_file).exists()}")
except Exception as e:
    print(f"✗ Failed: {e}")
print()

# Test 4: Default behavior without specifying output (creates _cropped file)
print("Test 4: Default behavior (creates _cropped suffix, no overwrite issue)")
try:
    # First time
    result1 = ImagenClient.autocrop_image(
        test_image,
        padding=0,
        overwrite=False,
    )
    print(f"✓ First run success: {result1}")

    # Second time should fail
    try:
        result2 = ImagenClient.autocrop_image(
            test_image,
            padding=0,
            overwrite=False,
        )
        print(f"✗ Second run unexpected success - should have failed!")
    except FileExistsError as e:
        print(f"✓ Correctly prevented overwrite on second run")

    # Third time with overwrite=True should succeed
    result3 = ImagenClient.autocrop_image(
        test_image,
        padding=0,
        overwrite=True,
    )
    print(f"✓ Third run with overwrite=True success")

except Exception as e:
    print(f"✗ Failed: {e}")
print()

print("All tests completed!")
print(f"Check output directory: {output_dir}")
