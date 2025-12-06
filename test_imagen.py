"""Quick test script for the Imagen client."""

import os
import subprocess
import tempfile
from pathlib import Path

from src.mcp_imagen_server.imagen_client import ImagenClient


def main():
    """Test image generation."""
    print("Testing Imagen client...")

    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Output directory: {tmpdir}")

        # Get project configuration
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        # Auto-detect project from gcloud if not set
        if not project:
            try:
                result = subprocess.run(
                    ["gcloud", "config", "get-value", "project"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    project = result.stdout.strip()
            except Exception:
                pass

        if not project:
            print("Error: No Google Cloud project configured.")
            print("Please set GOOGLE_CLOUD_PROJECT or configure gcloud default project.")
            return 1

        # Initialize client with Vertex AI
        print(f"Using Vertex AI with project: {project}, location: {location}")
        client = ImagenClient(project=project, location=location)

        # Test prompt
        prompt = "A cute robot holding a sign that says 'Hello MCP'"
        print(f"\nGenerating image with prompt: '{prompt}'")

        # Generate image
        try:
            file_paths = client.generate_images(
                prompt=prompt,
                model="imagen-4.0-fast-generate-001",
                output_dir=tmpdir,
                sample_count=1,
                aspect_ratio="1:1",
            )

            print(f"\n✓ Success! Generated {len(file_paths)} image(s):")
            for i, path in enumerate(file_paths, 1):
                file_size = Path(path).stat().st_size / 1024  # KB
                print(f"  {i}. {path} ({file_size:.1f} KB)")

            # Verify files exist and are not empty
            for path in file_paths:
                assert Path(path).exists(), f"File not found: {path}"
                assert Path(path).stat().st_size > 0, f"File is empty: {path}"

            print("\n✓ All tests passed!")
            return 0

        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback

            traceback.print_exc()
            return 1


if __name__ == "__main__":
    exit(main())
