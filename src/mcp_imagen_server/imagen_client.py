"""Google Imagen API client for image generation."""

import logging
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Supported models
ImagenModel = Literal[
    "imagen-4.0-generate-001",
    "imagen-4.0-fast-generate-001",
    "imagen-4.0-ultra-generate-001",
]

# Aspect ratios
AspectRatio = Literal["1:1", "3:4", "4:3", "9:16", "16:9"]


class ImagenClient:
    """Client for Google Imagen API image generation."""

    def __init__(
        self,
        vertexai: bool = False,
        project: str | None = None,
        location: str = "us-central1",
    ):
        """Initialize the Imagen client.

        Args:
            vertexai: Whether to use Vertex AI (True) or Gemini API (False)
            project: Google Cloud project ID (required for Vertex AI)
            location: Google Cloud location (default: us-central1)
        """
        if vertexai:
            if not project:
                raise ValueError("project is required when using Vertex AI")
            self.client = genai.Client(vertexai=True, project=project, location=location)
            logger.info(
                f"Initialized Imagen client with Vertex AI (project={project}, location={location})"
            )
        else:
            self.client = genai.Client()
            logger.info("Initialized Imagen client with Gemini API")

    def generate_images(
        self,
        prompt: str,
        model: ImagenModel = "imagen-4.0-generate-001",
        output_dir: str | Path = ".",
        sample_count: int = 1,
        aspect_ratio: AspectRatio = "1:1",
    ) -> list[str]:
        """Generate images from a text prompt.

        Args:
            prompt: Text description of the image to generate
            model: Imagen model to use
            output_dir: Directory to save generated images
            sample_count: Number of images to generate (1-4, always 1 for ultra model)
            aspect_ratio: Aspect ratio of generated images (default: 1:1)

        Returns:
            List of file paths to generated images

        Raises:
            ValueError: If parameters are invalid
            Exception: If image generation fails
        """
        # Validate parameters
        if sample_count < 1 or sample_count > 4:
            raise ValueError("sample_count must be between 1 and 4")

        if model == "imagen-4.0-ultra-generate-001" and sample_count != 1:
            raise ValueError("Ultra model only supports sample_count=1")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Generating {sample_count} image(s) with model={model}, aspect_ratio={aspect_ratio}"
        )
        logger.info(f"Prompt: {prompt}")

        try:
            # Generate images using Imagen API
            response = self.client.models.generate_images(
                model=model,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=sample_count,
                    output_mime_type="image/png",
                    aspect_ratio=aspect_ratio,
                ),
            )

            # Save generated images and collect file paths
            saved_files = []
            for i, generated_image in enumerate(response.generated_images):
                # Generate filename based on prompt and index
                safe_prompt = "".join(c if c.isalnum() or c in " -_" else "_" for c in prompt)
                safe_prompt = safe_prompt[:50]  # Limit length
                filename = f"{safe_prompt}_{i + 1}.png"
                filepath = output_path / filename

                # Get image bytes and save
                image_bytes = generated_image.image.image_bytes
                with open(filepath, "wb") as f:
                    f.write(image_bytes)

                saved_files.append(str(filepath.absolute()))
                logger.info(f"Saved image {i + 1}/{sample_count} to: {filepath}")

            logger.info(f"Successfully generated {len(saved_files)} image(s)")
            return saved_files

        except Exception as e:
            logger.error(f"Error generating images: {e}")
            raise
