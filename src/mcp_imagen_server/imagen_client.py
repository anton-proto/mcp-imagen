"""Google Imagen API client for image generation."""

import base64
import logging
from pathlib import Path
from typing import Literal

import httpx
from google import genai
from google.auth import default
from google.auth.transport.requests import Request
from google.genai import types
from PIL import Image
from rembg import remove

logger = logging.getLogger(__name__)

# Supported models
ImagenModel = Literal[
    "imagen-4.0-generate-001",
    "imagen-4.0-fast-generate-001",
    "imagen-4.0-ultra-generate-001",
]

# Customization model (Imagen 3 with style support)
CustomizationModel = Literal["imagen-3.0-capability-001"]

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
            self.project = project
            self.location = location
            self.vertexai = True
            logger.info(
                f"Initialized Imagen client with Vertex AI (project={project}, location={location})"
            )
        else:
            self.client = genai.Client()
            self.project = None
            self.location = None
            self.vertexai = False
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

    def generate_images_with_style(
        self,
        prompt: str,
        style_image_path: str | Path,
        style_description: str,
        output_dir: str | Path = ".",
        sample_count: int = 1,
    ) -> list[str]:
        """Generate images following the style of a reference image.

        Uses Imagen 3 Customization (imagen-3.0-capability-001) via REST API
        for pure text-to-image generation with style guidance.

        Args:
            prompt: Text description of the image to generate
            style_image_path: Path to the style reference image
            style_description: Description of the style
                (e.g., "watercolor style", "neon sign style")
            output_dir: Directory to save generated images
            sample_count: Number of images to generate (1-4)

        Returns:
            List of file paths to generated images

        Raises:
            ValueError: If parameters are invalid or Vertex AI not configured
            FileNotFoundError: If style image doesn't exist
            Exception: If image generation fails
        """
        # Validate Vertex AI is configured
        if not self.vertexai:
            raise ValueError(
                "Style customization requires Vertex AI. "
                "Initialize client with vertexai=True and provide project ID."
            )

        # Validate parameters
        if sample_count < 1 or sample_count > 4:
            raise ValueError("sample_count must be between 1 and 4")

        style_path = Path(style_image_path)
        if not style_path.exists():
            raise FileNotFoundError(f"Style image not found: {style_image_path}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating {sample_count} image(s) with style customization via REST API")
        logger.info(f"Style image: {style_image_path}")
        logger.info(f"Style description: {style_description}")
        logger.info(f"Prompt: {prompt}")

        try:
            # Read and base64 encode the style reference image
            with open(style_path, "rb") as f:
                style_image_bytes = f.read()
            style_image_b64 = base64.b64encode(style_image_bytes).decode("utf-8")

            # Build the full prompt with style reference
            full_prompt = (
                f"Generate an image in {style_description} [1] "
                f"based on the following caption: {prompt}"
            )

            # Build request body for Vertex AI REST API
            request_body = {
                "instances": [
                    {
                        "prompt": full_prompt,
                        "referenceImages": [
                            {
                                "referenceType": "REFERENCE_TYPE_STYLE",
                                "referenceId": 1,
                                "referenceImage": {"bytesBase64Encoded": style_image_b64},
                                "styleImageConfig": {"styleDescription": style_description},
                            }
                        ],
                    }
                ],
                "parameters": {"sampleCount": sample_count},
            }

            # Get access token
            credentials, _ = default()
            credentials.refresh(Request())
            access_token = credentials.token

            # Build API endpoint URL
            endpoint = (
                f"https://{self.location}-aiplatform.googleapis.com/v1/"
                f"projects/{self.project}/locations/{self.location}/"
                f"publishers/google/models/imagen-3.0-capability-001:predict"
            )

            # Make REST API call
            logger.info(f"Calling Vertex AI REST API: {endpoint}")
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                    },
                    json=request_body,
                )

            # Check for errors
            if response.status_code != 200:
                error_detail = response.text
                raise Exception(
                    f"API request failed with status {response.status_code}: {error_detail}"
                )

            # Parse response
            response_data = response.json()
            predictions = response_data.get("predictions", [])

            if not predictions:
                raise Exception("No images were generated in the response")

            # Save generated images and collect file paths
            saved_files = []
            for i, prediction in enumerate(predictions):
                # Get base64 image data
                image_b64 = prediction.get("bytesBase64Encoded")
                if not image_b64:
                    logger.warning(f"Prediction {i} missing image data, skipping")
                    continue

                # Decode image
                image_bytes = base64.b64decode(image_b64)

                # Generate filename
                safe_prompt = "".join(c if c.isalnum() or c in " -_" else "_" for c in prompt)
                safe_prompt = safe_prompt[:40]  # Limit length
                safe_style = "".join(
                    c if c.isalnum() or c in " -_" else "_" for c in style_description
                )
                safe_style = safe_style[:20]
                filename = f"{safe_prompt}_style_{safe_style}_{i + 1}.png"
                filepath = output_path / filename

                # Save image
                with open(filepath, "wb") as f:
                    f.write(image_bytes)

                saved_files.append(str(filepath.absolute()))
                logger.info(f"Saved styled image {i + 1}/{len(predictions)} to: {filepath}")

            if not saved_files:
                raise Exception("No images could be saved from the response")

            logger.info(f"Successfully generated {len(saved_files)} styled image(s)")
            return saved_files

        except Exception as e:
            logger.error(f"Error generating styled images: {e}")
            raise

    @staticmethod
    def remove_background(
        input_path: str | Path,
        output_path: str | Path | None = None,
    ) -> str:
        """Remove background from an image using rembg.

        Args:
            input_path: Path to the input image file
            output_path: Path to save the output image (optional).
                If not provided, will save with 'nobg_' prefix in same directory.

        Returns:
            Path to the output image file with background removed

        Raises:
            FileNotFoundError: If input image doesn't exist
            Exception: If background removal fails
        """
        # Validate input path
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")

        # Determine output path
        if output_path is None:
            output_file = input_file.parent / f"nobg_{input_file.name}"
        else:
            output_file = Path(output_path)

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Removing background from: {input_path}")
        logger.info(f"Output will be saved to: {output_file}")

        try:
            # Open input image
            with open(input_file, "rb") as f:
                input_image = f.read()

            # Remove background
            output_image = remove(input_image)

            # Save output image
            with open(output_file, "wb") as f:
                f.write(output_image)

            logger.info(f"Successfully removed background and saved to: {output_file}")
            return str(output_file.absolute())

        except Exception as e:
            logger.error(f"Error removing background: {e}")
            raise

    @staticmethod
    def autocrop_image(
        input_path: str | Path,
        output_path: str | Path | None = None,
        padding: int = 0,
        overwrite: bool = True,
    ) -> str:
        """Automatically crop an image to remove transparent or empty borders.

        Args:
            input_path: Path to the input image file
            output_path: Path to save the cropped image (optional).
                If not provided, will save with '_cropped' suffix in same directory.
            padding: Number of pixels to add as padding around cropped content (default: 0)
            overwrite: Whether to overwrite existing output files (default: True).
                If False and output file exists, raises FileExistsError.

        Returns:
            Path to the output cropped image file

        Raises:
            FileNotFoundError: If input image doesn't exist
            FileExistsError: If output file exists and overwrite=False
            ValueError: If image is completely transparent or padding is negative
            Exception: If cropping fails
        """
        # Validate input path
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")

        # Validate padding
        if padding < 0:
            raise ValueError("padding must be non-negative")

        # Determine output path
        if output_path is None:
            output_file = (
                input_file.parent / f"{input_file.stem}_cropped{input_file.suffix}"
            )
        else:
            output_file = Path(output_path)

        # Check if output file exists and handle overwrite
        if output_file.exists() and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {output_file}. "
                "Use overwrite=True to overwrite existing files."
            )

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Auto-cropping image: {input_path}")
        if padding > 0:
            logger.info(f"Using padding: {padding}px")
        if output_file.exists():
            logger.info(f"Overwriting existing file: {output_file}")

        try:
            # Open the image
            image = Image.open(input_file)

            # Convert to RGBA if not already (to handle transparency)
            if image.mode != "RGBA":
                image = image.convert("RGBA")

            # Get the bounding box of non-transparent pixels
            bbox = image.getbbox()

            if bbox is None:
                raise ValueError(
                    "Image appears to be completely transparent or empty - cannot autocrop"
                )

            # Add padding if specified
            if padding > 0:
                left, upper, right, lower = bbox
                width, height = image.size

                # Ensure padding doesn't go outside image bounds
                left = max(0, left - padding)
                upper = max(0, upper - padding)
                right = min(width, right + padding)
                lower = min(height, lower + padding)

                bbox = (left, upper, right, lower)

            # Crop the image
            cropped = image.crop(bbox)

            # Save the cropped image
            cropped.save(output_file)

            # Log dimensions
            original_size = image.size
            cropped_size = cropped.size
            logger.info(f"Original size: {original_size[0]}x{original_size[1]}")
            logger.info(f"Cropped size:  {cropped_size[0]}x{cropped_size[1]}")
            logger.info(f"Successfully auto-cropped and saved to: {output_file}")

            return str(output_file.absolute())

        except Exception as e:
            logger.error(f"Error auto-cropping image: {e}")
            raise
