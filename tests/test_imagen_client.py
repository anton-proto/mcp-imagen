"""Unit tests for Imagen client."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.mcp_imagen_server.imagen_client import ImagenClient


class TestImagenClient:
    """Test suite for ImagenClient."""

    def test_client_initialization_gemini(self):
        """Test initializing client with Gemini API."""
        with patch("src.mcp_imagen_server.imagen_client.genai.Client") as mock_client:
            client = ImagenClient(vertexai=False)
            assert client.vertexai is False
            assert client.project is None
            assert client.location is None
            mock_client.assert_called_once_with()

    def test_client_initialization_vertexai(self):
        """Test initializing client with Vertex AI."""
        with patch("src.mcp_imagen_server.imagen_client.genai.Client") as mock_client:
            client = ImagenClient(vertexai=True, project="test-project", location="us-central1")
            assert client.vertexai is True
            assert client.project == "test-project"
            assert client.location == "us-central1"
            mock_client.assert_called_once_with(
                vertexai=True, project="test-project", location="us-central1"
            )

    def test_generate_images_creates_directory(self):
        """Test that generate_images creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "new_subdir"
            assert not output_dir.exists()

            with patch("src.mcp_imagen_server.imagen_client.genai.Client") as mock_client_class:
                # Mock the client and response
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_image = MagicMock()
                mock_image.image.image_bytes = b"fake image data"
                mock_response.generated_images = [mock_image]
                mock_client.models.generate_images.return_value = mock_response
                mock_client_class.return_value = mock_client

                client = ImagenClient(vertexai=False)
                try:
                    client.generate_images(
                        prompt="test prompt", output_dir=str(output_dir), sample_count=1
                    )
                    assert output_dir.exists()
                except Exception:
                    # Even if generation fails, directory should be created
                    assert output_dir.exists()

    def test_generate_images_validates_sample_count(self):
        """Test that sample_count validation works correctly."""
        with patch("src.mcp_imagen_server.imagen_client.genai.Client"):
            client = ImagenClient(vertexai=False)

            # Test ultra model with sample_count > 1
            with pytest.raises(ValueError, match="Ultra model only supports sample_count=1"):
                client.generate_images(
                    prompt="test",
                    model="imagen-4.0-ultra-generate-001",
                    output_dir="/tmp",
                    sample_count=2,
                )

            # Test sample_count out of range
            with pytest.raises(ValueError, match="sample_count must be between 1 and 4"):
                client.generate_images(
                    prompt="test",
                    model="imagen-4.0-generate-001",
                    output_dir="/tmp",
                    sample_count=5,
                )

    def test_generate_images_accepts_valid_aspect_ratio(self):
        """Test that generate_images accepts valid aspect ratios."""
        with patch("src.mcp_imagen_server.imagen_client.genai.Client") as mock_client_class:
            # Mock the client and response
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.generated_images = []
            mock_client.models.generate_images.return_value = mock_response
            mock_client_class.return_value = mock_client

            client = ImagenClient(vertexai=False)

            # Valid aspect ratios should not raise errors (though generation may fail)
            valid_ratios = ["1:1", "3:4", "4:3", "9:16", "16:9"]
            for ratio in valid_ratios:
                try:
                    client.generate_images(prompt="test", output_dir="/tmp", aspect_ratio=ratio)
                except Exception as e:
                    # Ignore other exceptions, we're just testing validation doesn't raise
                    if "Invalid aspect_ratio" in str(e):
                        pytest.fail(f"Should not raise validation error for valid ratio: {ratio}")

    def test_generate_images_with_style_requires_vertexai(self):
        """Test that generate_images_with_style requires Vertex AI."""
        with patch("src.mcp_imagen_server.imagen_client.genai.Client"):
            client = ImagenClient(vertexai=False)

            with pytest.raises(ValueError, match="Style customization requires Vertex AI"):
                client.generate_images_with_style(
                    prompt="test",
                    style_image_path="/nonexistent/file.png",
                    style_description="test style",
                    output_dir="/tmp",
                )

    def test_generate_images_with_style_validates_files(self):
        """Test that generate_images_with_style validates input files when using Vertex AI."""
        with patch("src.mcp_imagen_server.imagen_client.genai.Client"):
            client = ImagenClient(vertexai=True, project="test-project")

            with pytest.raises(FileNotFoundError, match="Style image not found"):
                client.generate_images_with_style(
                    prompt="test",
                    style_image_path="/nonexistent/file.png",
                    style_description="test style",
                    output_dir="/tmp",
                )


@pytest.mark.skipif(
    not os.getenv("GOOGLE_GENAI_API_KEY") and not os.getenv("USE_VERTEXAI"),
    reason="Google API credentials not configured",
)
class TestImagenClientIntegration:
    """Integration tests that require actual API credentials."""

    def test_generate_images_integration(self):
        """Integration test for image generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            use_vertexai = os.getenv("USE_VERTEXAI", "false").lower() == "true"
            project = os.getenv("GOOGLE_CLOUD_PROJECT", "wired-balm-187912")

            if use_vertexai:
                client = ImagenClient(vertexai=True, project=project)
            else:
                client = ImagenClient(vertexai=False)

            file_paths = client.generate_images(
                prompt="A simple red circle",
                model="imagen-4.0-fast-generate-001",
                output_dir=tmpdir,
                sample_count=1,
                aspect_ratio="1:1",
            )

            assert len(file_paths) == 1
            assert Path(file_paths[0]).exists()
            assert Path(file_paths[0]).stat().st_size > 0
