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

        # Detect authentication method automatically
        api_key = os.getenv("GOOGLE_API_KEY")
        use_vertexai = os.getenv("USE_VERTEXAI", "").lower() == "true"
        project = os.getenv("GOOGLE_CLOUD_PROJECT")

        # Auto-detect project from gcloud if not set
        if not project and not api_key:
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

        # Initialize client based on available credentials
        if use_vertexai or (not api_key and project):
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            print(f"Using Vertex AI with project: {project}, location: {location}")
            client = ImagenClient(vertexai=True, project=project, location=location)
        else:
            if api_key:
                print("Using Gemini API with API key")
            else:
                print("Using Gemini API with ADC")
            client = ImagenClient(vertexai=False, api_key=api_key)

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
