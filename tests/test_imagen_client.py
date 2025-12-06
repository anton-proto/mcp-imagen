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

    def test_remove_background_single_image(self):
        """Test removing background from a single image (preserve mode)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy input image
            input_file = Path(tmpdir) / "test_image.png"
            input_file.write_bytes(b"fake image data")

            with patch("src.mcp_imagen_server.imagen_client.remove") as mock_remove:
                mock_remove.return_value = b"fake output data"

                result = ImagenClient.remove_background(
                    input_paths=str(input_file), overwrite=False
                )

                assert "input" in result
                assert "output" in result
                assert Path(result["input"]).exists()
                assert Path(result["output"]).exists()
                assert "nobg_" in result["output"]

    def test_remove_background_single_with_output_dir(self):
        """Test removing background from single image with output_dir specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create input and output directories
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            input_file = input_dir / "test.png"
            input_file.write_bytes(b"fake image data")

            with patch("src.mcp_imagen_server.imagen_client.remove") as mock_remove:
                mock_remove.return_value = b"fake output data"

                result = ImagenClient.remove_background(
                    input_paths=str(input_file), output_dir=str(output_dir), overwrite=False
                )

                assert "output" in result
                output_path = Path(result["output"])
                assert output_path.parent == output_dir
                assert output_path.exists()

    def test_remove_background_batch_processing(self):
        """Test batch background removal with multiple images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple input images
            input_files = []
            for i in range(3):
                input_file = Path(tmpdir) / f"test_{i}.png"
                input_file.write_bytes(b"fake image data")
                input_files.append(str(input_file))

            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            with patch("src.mcp_imagen_server.imagen_client.remove") as mock_remove:
                mock_remove.return_value = b"fake output data"

                result = ImagenClient.remove_background(
                    input_paths=input_files, output_dir=str(output_dir), overwrite=False
                )

                assert "results" in result
                assert "successful" in result
                assert "failed" in result
                assert result["successful"] == 3
                assert result["failed"] == 0
                assert len(result["results"]) == 3

                # Verify all output files exist
                for item in result["results"]:
                    assert item["error"] is None
                    assert Path(item["output"]).exists()
                    assert Path(item["output"]).parent == output_dir

    def test_remove_background_batch_handles_errors(self):
        """Test that batch processing handles individual file errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid and invalid input paths
            valid_file = Path(tmpdir) / "valid.png"
            valid_file.write_bytes(b"fake image data")

            invalid_file = Path(tmpdir) / "nonexistent.png"
            # Don't create this file

            input_files = [str(valid_file), str(invalid_file)]
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            with patch("src.mcp_imagen_server.imagen_client.remove") as mock_remove:
                mock_remove.return_value = b"fake output data"

                result = ImagenClient.remove_background(
                    input_paths=input_files, output_dir=str(output_dir)
                )

                assert result["successful"] == 1
                assert result["failed"] == 1
                assert len(result["results"]) == 2

                # Check that one succeeded and one failed
                errors = [item["error"] for item in result["results"]]
                assert None in errors  # One succeeded
                assert any(e is not None for e in errors)  # One failed

    def test_remove_background_batch_requires_output_dir_when_preserve(self):
        """Test that batch processing requires output_dir when overwrite=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_files = [str(Path(tmpdir) / "test.png")]

            with pytest.raises(ValueError, match="output_dir is required"):
                ImagenClient.remove_background(input_paths=input_files, overwrite=False)

    def test_remove_background_batch_parallel_execution(self):
        """Test that batch processing uses parallel execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple input images
            input_files = []
            for i in range(5):
                input_file = Path(tmpdir) / f"test_{i}.png"
                input_file.write_bytes(b"fake image data")
                input_files.append(str(input_file))

            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            with patch("src.mcp_imagen_server.imagen_client.remove") as mock_remove:
                mock_remove.return_value = b"fake output data"

                # Test with different worker counts
                result = ImagenClient.remove_background(
                    input_paths=input_files,
                    output_dir=str(output_dir),
                    overwrite=False,
                    max_workers=2,
                )

                assert result["successful"] == 5
                assert len(result["results"]) == 5

    def test_remove_background_overwrite_single(self):
        """Test overwrite mode for single image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy input image
            input_file = Path(tmpdir) / "test_image.png"
            input_file.write_bytes(b"fake image data")

            with patch("src.mcp_imagen_server.imagen_client.remove") as mock_remove:
                mock_remove.return_value = b"fake output data"

                result = ImagenClient.remove_background(input_paths=str(input_file), overwrite=True)

                assert "input" in result
                assert "output" in result
                # In overwrite mode, output should be same as input
                assert result["output"] == str(input_file.absolute())
                # File should contain the processed data
                assert input_file.read_bytes() == b"fake output data"

    def test_remove_background_overwrite_batch(self):
        """Test overwrite mode for batch processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple input images
            input_files = []
            for i in range(3):
                input_file = Path(tmpdir) / f"test_{i}.png"
                input_file.write_bytes(b"fake image data")
                input_files.append(str(input_file))

            with patch("src.mcp_imagen_server.imagen_client.remove") as mock_remove:
                mock_remove.return_value = b"fake output data"

                result = ImagenClient.remove_background(input_paths=input_files, overwrite=True)

                assert result["successful"] == 3
                assert result["failed"] == 0

                # Verify all files were overwritten
                for item in result["results"]:
                    assert item["error"] is None
                    # In overwrite mode, output equals input
                    assert item["output"] == item["input"]
                    # Original files should be overwritten
                    assert Path(item["input"]).read_bytes() == b"fake output data"

    def test_remove_background_preserve_mode(self):
        """Test preserve mode (overwrite=False) for batch processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple input images
            input_files = []
            original_data = b"original image data"
            for i in range(3):
                input_file = Path(tmpdir) / f"test_{i}.png"
                input_file.write_bytes(original_data)
                input_files.append(str(input_file))

            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            with patch("src.mcp_imagen_server.imagen_client.remove") as mock_remove:
                mock_remove.return_value = b"processed data"

                result = ImagenClient.remove_background(
                    input_paths=input_files, output_dir=str(output_dir), overwrite=False
                )

                assert result["successful"] == 3

                # Verify original files are unchanged
                for input_path in input_files:
                    assert Path(input_path).read_bytes() == original_data

                # Verify output files exist with processed data
                for item in result["results"]:
                    output_path = Path(item["output"])
                    assert output_path.exists()
                    assert output_path.read_bytes() == b"processed data"
                    assert output_path.parent == output_dir
                    assert "nobg_" in output_path.name

    def test_remove_background_requires_output_dir_when_preserve(self):
        """Test that output_dir is required for batch when overwrite=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_files = [str(Path(tmpdir) / "test.png")]

            with pytest.raises(ValueError, match="output_dir is required"):
                ImagenClient.remove_background(input_paths=input_files, overwrite=False)

    def test_generate_images_batch_prompts(self):
        """Test batch image generation with multiple prompts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

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

                prompts = ["prompt 1", "prompt 2", "prompt 3"]
                result = client.generate_images(prompt=prompts, output_dir=str(output_dir))

                assert "results" in result
                assert "successful" in result
                assert "failed" in result
                assert result["successful"] == 3
                assert result["failed"] == 0
                assert len(result["results"]) == 3

                # Verify each result has correct structure
                for item in result["results"]:
                    assert "prompt" in item
                    assert "files" in item
                    assert "error" in item
                    assert item["error"] is None
                    assert len(item["files"]) == 1

    def test_generate_images_from_prompt_files(self):
        """Test batch image generation from prompt files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create prompt files
            prompt_files = []
            for i in range(2):
                prompt_file = Path(tmpdir) / f"prompt_{i}.txt"
                prompt_file.write_text(f"Test prompt {i}")
                prompt_files.append(str(prompt_file))

            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

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

                result = client.generate_images(
                    prompt_files=prompt_files, output_dir=str(output_dir)
                )

                assert result["successful"] == 2
                assert len(result["results"]) == 2

    def test_generate_images_batch_with_output_dirs(self):
        """Test batch generation with separate output directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create output directories
            output_dirs = []
            for i in range(2):
                out_dir = Path(tmpdir) / f"output_{i}"
                out_dir.mkdir()
                output_dirs.append(str(out_dir))

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

                prompts = ["prompt 1", "prompt 2"]
                result = client.generate_images(
                    prompt=prompts, output_dir=".", output_dirs=output_dirs
                )

                assert result["successful"] == 2
                # Verify files are in one of the specified output directories
                # (order may vary due to parallel processing)
                output_dir_set = {Path(d) for d in output_dirs}
                for item in result["results"]:
                    for file_path in item["files"]:
                        assert Path(file_path).parent in output_dir_set

    def test_generate_images_batch_handles_errors(self):
        """Test that batch generation handles individual prompt errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with patch("src.mcp_imagen_server.imagen_client.genai.Client") as mock_client_class:
                # Mock the client to fail on second call
                mock_client = MagicMock()
                mock_response = MagicMock()
                mock_image = MagicMock()
                mock_image.image.image_bytes = b"fake image data"
                mock_response.generated_images = [mock_image]

                # First call succeeds, second fails
                call_count = 0

                def side_effect(*args, **kwargs):
                    nonlocal call_count
                    call_count += 1
                    if call_count == 2:
                        raise Exception("API error")
                    return mock_response

                mock_client.models.generate_images.side_effect = side_effect
                mock_client_class.return_value = mock_client

                client = ImagenClient(vertexai=False)

                prompts = ["prompt 1", "prompt 2"]
                result = client.generate_images(prompt=prompts, output_dir=str(output_dir))

                assert result["successful"] == 1
                assert result["failed"] == 1
                assert len(result["results"]) == 2

                # Check that one succeeded and one failed
                errors = [item["error"] for item in result["results"]]
                assert None in errors  # One succeeded
                assert any(e is not None for e in errors)  # One failed


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
