"""Google Imagen API client for image generation."""

import base64
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import httpx
from google import genai
from google.auth import default
from google.auth.transport.requests import Request
from google.genai import types
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

    def _generate_images_single(
        self,
        prompt: str,
        model: ImagenModel,
        output_dir: Path,
        sample_count: int,
        aspect_ratio: AspectRatio,
    ) -> tuple[str, list[str], str | None]:
        """Generate images from a single prompt.

        Args:
            prompt: Text description of the image to generate
            model: Imagen model to use
            output_dir: Directory to save generated images
            sample_count: Number of images to generate (1-4, always 1 for ultra model)
            aspect_ratio: Aspect ratio of generated images

        Returns:
            Tuple of (prompt, file_paths, error_message)
            error_message is None if successful
        """
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
                filepath = output_dir / filename

                # Get image bytes and save
                image_bytes = generated_image.image.image_bytes
                with open(filepath, "wb") as f:
                    f.write(image_bytes)

                saved_files.append(str(filepath.absolute()))
                logger.info(f"Saved image {i + 1}/{sample_count} to: {filepath}")

            return prompt, saved_files, None

        except Exception as e:
            error_msg = f"Error generating images: {e}"
            logger.error(error_msg)
            return prompt, [], str(e)

    def generate_images(
        self,
        prompt: str | list[str] | None = None,
        prompt_files: list[str | Path] | None = None,
        model: ImagenModel = "imagen-4.0-generate-001",
        output_dir: str | Path = ".",
        output_dirs: list[str | Path] | None = None,
        sample_count: int = 1,
        aspect_ratio: AspectRatio = "1:1",
        max_workers: int = 4,
    ) -> dict[str, any] | list[str]:
        """Generate images from text prompt(s).

        Args:
            prompt: Single prompt or list of prompts
            prompt_files: List of files containing prompts (one prompt per file)
            model: Imagen model to use
            output_dir: Directory for single prompt or batch with same output
            output_dirs: List of output directories (one per prompt for batch)
            sample_count: Number of images to generate per prompt (1-4, always 1 for ultra)
            aspect_ratio: Aspect ratio of generated images (default: 1:1)
            max_workers: Maximum parallel workers for batch processing (default: 4)

        Returns:
            For single prompt: List of file paths
            For batch: {
                "results": [{"prompt": str, "files": list, "error": str | None}, ...],
                "successful": int,
                "failed": int
            }

        Raises:
            ValueError: If parameters are invalid
            Exception: If image generation fails (single prompt mode only)
        """
        # Validate parameters
        if sample_count < 1 or sample_count > 4:
            raise ValueError("sample_count must be between 1 and 4")

        if model == "imagen-4.0-ultra-generate-001" and sample_count != 1:
            raise ValueError("Ultra model only supports sample_count=1")

        # Load prompts from files if specified
        prompts_from_files = []
        if prompt_files:
            for prompt_file in prompt_files:
                file_path = Path(prompt_file)
                if not file_path.exists():
                    raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
                with open(file_path, encoding="utf-8") as f:
                    file_prompt = f.read().strip()
                    if file_prompt:
                        prompts_from_files.append(file_prompt)

        # Determine prompts to process
        if prompt and isinstance(prompt, str):
            # Single prompt mode
            if prompt_files:
                raise ValueError("Cannot specify both prompt and prompt_files")
            if output_dirs:
                raise ValueError("output_dirs not supported for single prompt")

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"Generating {sample_count} image(s) with "
                f"model={model}, aspect_ratio={aspect_ratio}"
            )
            logger.info(f"Prompt: {prompt}")

            _, saved_files, error = self._generate_images_single(
                prompt, model, output_path, sample_count, aspect_ratio
            )
            if error:
                raise Exception(error)

            logger.info(f"Successfully generated {len(saved_files)} image(s)")
            return saved_files

        # Batch mode - collect all prompts
        elif prompt and isinstance(prompt, list):
            prompts_list = prompt
        elif prompts_from_files:
            prompts_list = prompts_from_files
        else:
            raise ValueError("Either prompt or prompt_files must be provided")

        # Validate batch mode
        if not prompts_list:
            raise ValueError("No prompts to process")

        # Determine output directories for batch
        if output_dirs:
            if len(output_dirs) != len(prompts_list):
                raise ValueError(
                    f"output_dirs length ({len(output_dirs)}) must match "
                    f"number of prompts ({len(prompts_list)})"
                )
            output_paths = [Path(d) for d in output_dirs]
        else:
            # Use single output_dir for all
            output_paths = [Path(output_dir)] * len(prompts_list)

        # Create all output directories
        for out_path in set(output_paths):
            out_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting batch image generation for {len(prompts_list)} prompts")
        logger.info(f"Using {max_workers} parallel workers")

        results = []
        successful = 0
        failed = 0

        # Process prompts in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_prompt = {}
            for prompt_text, out_path in zip(prompts_list, output_paths, strict=False):
                future = executor.submit(
                    self._generate_images_single,
                    prompt_text,
                    model,
                    out_path,
                    sample_count,
                    aspect_ratio,
                )
                future_to_prompt[future] = (prompt_text, out_path)

            # Collect results as they complete
            for future in as_completed(future_to_prompt):
                try:
                    prompt_text, saved_files, error = future.result()
                    if error:
                        failed += 1
                        results.append(
                            {"prompt": prompt_text, "files": saved_files, "error": error}
                        )
                    else:
                        successful += 1
                        results.append({"prompt": prompt_text, "files": saved_files, "error": None})
                except Exception as e:
                    failed += 1
                    prompt_text, _ = future_to_prompt[future]
                    results.append({"prompt": prompt_text, "files": [], "error": str(e)})

        logger.info(f"Batch processing complete: {successful} successful, {failed} failed")

        return {"results": results, "successful": successful, "failed": failed}

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
    def _remove_background_single(
        input_path: str | Path,
        output_path: str | Path | None = None,
    ) -> tuple[str, str, str | None]:
        """Remove background from a single image.

        Args:
            input_path: Path to the input image file
            output_path: Path to save the output image (optional).
                If not provided, will save with 'nobg_' prefix in same directory.

        Returns:
            Tuple of (input_path, output_path, error_message)
            error_message is None if successful

        Raises:
            FileNotFoundError: If input image doesn't exist
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
            return str(input_file.absolute()), str(output_file.absolute()), None

        except Exception as e:
            error_msg = f"Error removing background: {e}"
            logger.error(error_msg)
            return str(input_file.absolute()), "", str(e)

    @staticmethod
    def remove_background(
        input_paths: str | Path | list[str | Path],
        output_dir: str | Path | None = None,
        overwrite: bool = True,
        max_workers: int = 4,
    ) -> dict[str, str | list[dict[str, str]]]:
        """Remove background from one or more images using rembg.

        Args:
            input_paths: Single path or list of paths to input image files
            output_dir: Directory to save output images (optional).
                If not provided for single image, saves with 'nobg_' prefix in same directory.
                For batch processing with overwrite=False, output_dir is required.
            overwrite: If True, replace original images with background-removed versions.
                If False, save to output_dir with 'nobg_' prefix (default: True)
            max_workers: Maximum number of parallel workers for batch processing (default: 4)

        Returns:
            For single image: {"input": str, "output": str}
            For batch: {
                "results": [{"input": str, "output": str, "error": str | None}, ...],
                "successful": int,
                "failed": int
            }

        Raises:
            ValueError: If batch processing with overwrite=False without output_dir
            FileNotFoundError: If input image doesn't exist (single image mode)
        """
        # Handle single image case
        if isinstance(input_paths, (str, Path)):
            input_path = input_paths
            input_file = Path(input_path)

            if overwrite:
                # Overwrite mode: replace original file
                output_path = input_path
            elif output_dir:
                # Non-overwrite with output_dir: save to output_dir with prefix
                output_dir_path = Path(output_dir)
                output_path = output_dir_path / f"nobg_{input_file.name}"
            else:
                # Non-overwrite without output_dir: save with prefix in same directory
                output_path = None

            input_abs, output_abs, error = ImagenClient._remove_background_single(
                input_path, output_path
            )
            if error:
                raise Exception(error)
            return {"input": input_abs, "output": output_abs}

        # Handle batch processing
        if not overwrite and not output_dir:
            raise ValueError(
                "output_dir is required for batch background removal when overwrite=False"
            )

        logger.info(f"Starting batch background removal for {len(input_paths)} images")
        logger.info(f"Overwrite mode: {overwrite}")
        logger.info(f"Using {max_workers} parallel workers")

        results = []
        successful = 0
        failed = 0

        # Process images in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_input = {}
            for input_path in input_paths:
                input_file = Path(input_path)

                if overwrite:
                    # Overwrite mode: replace original file
                    output_path = input_path
                else:
                    # Non-overwrite mode: save to output_dir with prefix
                    output_dir_path = Path(output_dir)
                    output_dir_path.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir_path / f"nobg_{input_file.name}"

                future = executor.submit(
                    ImagenClient._remove_background_single, input_path, output_path
                )
                future_to_input[future] = input_path

            # Collect results as they complete
            for future in as_completed(future_to_input):
                try:
                    input_abs, output_abs, error = future.result()
                    if error:
                        failed += 1
                        results.append({"input": input_abs, "output": output_abs, "error": error})
                    else:
                        successful += 1
                        results.append({"input": input_abs, "output": output_abs, "error": None})
                except Exception as e:
                    failed += 1
                    input_path = future_to_input[future]
                    results.append(
                        {"input": str(Path(input_path).absolute()), "output": "", "error": str(e)}
                    )

        logger.info(f"Batch processing complete: {successful} successful, {failed} failed")

        return {"results": results, "successful": successful, "failed": failed}
