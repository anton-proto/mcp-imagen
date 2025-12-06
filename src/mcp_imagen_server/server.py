"""MCP server for Google Imagen API."""

import asyncio
import logging
import os
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .imagen_client import AspectRatio, ImagenClient, ImagenModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

# Initialize MCP server
server = Server("mcp-imagen-server")

# Initialize Imagen client (will be set in main)
imagen_client: ImagenClient | None = None


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="text-to-image",
            description=(
                "Generate images from text prompt(s) using Google Imagen API. "
                "Supports single prompt, multiple prompts, or prompts from files. "
                "Returns paths to generated PNG files."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Single text prompt (for single image generation)",
                    },
                    "prompts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of text prompts (for batch processing)",
                    },
                    "prompt_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Array of file paths containing prompts "
                            "(one prompt per file, for batch processing)"
                        ),
                    },
                    "output_dir": {
                        "type": "string",
                        "description": (
                            "Absolute path to directory for saving images. "
                            "Used for single prompt or as default for batch."
                        ),
                    },
                    "output_dirs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Array of output directories (one per prompt for batch). "
                            "If not specified, output_dir is used for all."
                        ),
                    },
                    "model": {
                        "type": "string",
                        "enum": [
                            "imagen-4.0-generate-001",
                            "imagen-4.0-fast-generate-001",
                            "imagen-4.0-ultra-generate-001",
                        ],
                        "description": (
                            "Imagen model to use. Default: imagen-4.0-generate-001. "
                            "Ultra model only supports sample_count=1."
                        ),
                        "default": "imagen-4.0-generate-001",
                    },
                    "sample_count": {
                        "type": "integer",
                        "description": (
                            "Number of images per prompt (1-4). Must be 1 for ultra model. "
                            "Default: 1"
                        ),
                        "minimum": 1,
                        "maximum": 4,
                        "default": 1,
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "enum": ["1:1", "3:4", "4:3", "9:16", "16:9"],
                        "description": "Aspect ratio of generated images. Default: 1:1",
                        "default": "1:1",
                    },
                    "max_workers": {
                        "type": "integer",
                        "description": "Maximum parallel workers for batch processing (default: 4)",
                        "minimum": 1,
                        "maximum": 16,
                        "default": 4,
                    },
                },
            },
        ),
        Tool(
            name="style-to-image",
            description=(
                "Generate images following the style of a reference image using "
                "Imagen 3 Customization. Provide a style reference image and the model "
                "will generate new images matching that style. "
                "Returns paths to generated PNG files."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image content to generate",
                    },
                    "style_image_path": {
                        "type": "string",
                        "description": "Absolute path to the style reference image file",
                    },
                    "style_description": {
                        "type": "string",
                        "description": (
                            "Description of the style in the reference image "
                            "(e.g., 'watercolor painting style', 'neon sign style', 'mosaic style')"
                        ),
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Absolute path to directory where images should be saved",
                    },
                    "sample_count": {
                        "type": "integer",
                        "description": "Number of images to generate (1-4). Default: 1",
                        "minimum": 1,
                        "maximum": 4,
                        "default": 1,
                    },
                },
                "required": ["prompt", "style_image_path", "style_description", "output_dir"],
            },
        ),
        Tool(
            name="remove-background",
            description=(
                "Remove background from one or more images using rembg AI model. "
                "Supports both single image and batch processing with parallel execution. "
                "Returns path(s) to output image(s) with transparent background."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": (
                            "Absolute path to a single input image file "
                            "(for single image processing)"
                        ),
                    },
                    "input_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Array of absolute paths to input image files (for batch processing)"
                        ),
                    },
                    "output_dir": {
                        "type": "string",
                        "description": (
                            "Absolute path to directory for saving output images. "
                            "Required when overwrite=False. "
                            "Images saved with 'nobg_' prefix when overwrite=False."
                        ),
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": (
                            "If true, replace original images with background-removed versions. "
                            "If false, save to output_dir with 'nobg_' prefix. "
                            "Default: true"
                        ),
                        "default": True,
                    },
                    "max_workers": {
                        "type": "integer",
                        "description": "Maximum parallel workers for batch processing (default: 4)",
                        "minimum": 1,
                        "maximum": 16,
                        "default": 4,
                    },
                },
            },
        ),
        Tool(
            name="autocrop",
            description=(
                "Automatically crop images to remove transparent or empty borders. "
                "Supports single or batch processing with parallel execution. "
                "Returns paths to cropped images."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "input_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of absolute paths to input image files to crop. "
                            "Can be a single file or multiple files for batch processing."
                        ),
                    },
                    "output_dir": {
                        "type": "string",
                        "description": (
                            "Absolute path to output directory (optional). "
                            "If not provided and overwrite=True, input files will be overwritten. "
                            "If not provided and overwrite=False, files will be saved in the same "
                            "directory with '_cropped' suffix."
                        ),
                    },
                    "padding": {
                        "type": "integer",
                        "description": (
                            "Number of pixels to add as padding around cropped content (default: 0)"
                        ),
                        "minimum": 0,
                        "default": 0,
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": (
                            "Whether to overwrite existing output files (default: True). "
                            "If False and output file exists, "
                            "the operation will fail with an error."
                        ),
                        "default": True,
                    },
                },
                "required": ["input_paths"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name not in ["text-to-image", "style-to-image", "remove-background", "autocrop"]:
        raise ValueError(f"Unknown tool: {name}")

    if not imagen_client and name not in ["remove-background", "autocrop"]:
        raise RuntimeError("Imagen client not initialized")

    try:
        if name == "text-to-image":
            # Extract parameters for text-to-image
            prompt = arguments.get("prompt")
            prompts = arguments.get("prompts")
            prompt_files = arguments.get("prompt_files")
            output_dir = arguments.get("output_dir")
            output_dirs = arguments.get("output_dirs")
            model: ImagenModel = arguments.get("model", "imagen-4.0-generate-001")
            sample_count = arguments.get("sample_count", 1)
            aspect_ratio: AspectRatio = arguments.get("aspect_ratio", "1:1")
            max_workers = arguments.get("max_workers", 4)

            # Validate at least one prompt source
            if not prompt and not prompts and not prompt_files:
                raise ValueError("One of prompt, prompts, or prompt_files must be provided")

            # Validate output_dir
            if output_dir:
                output_path = Path(output_dir)
                if not output_path.is_absolute():
                    raise ValueError(f"output_dir must be an absolute path, got: {output_dir}")
            elif not output_dirs:
                raise ValueError("Either output_dir or output_dirs must be provided")

            # Validate output_dirs if provided
            if output_dirs:
                for out_dir in output_dirs:
                    if not Path(out_dir).is_absolute():
                        raise ValueError(f"All output_dirs must be absolute, got: {out_dir}")

            # Validate prompt_files if provided
            if prompt_files:
                for file_path in prompt_files:
                    if not Path(file_path).is_absolute():
                        raise ValueError(f"All prompt_files must be absolute, got: {file_path}")

            # Handle single prompt mode
            if prompt:
                logger.info(f"Generating images with prompt: {prompt[:100]}...")

                result = imagen_client.generate_images(
                    prompt=prompt,
                    model=model,
                    output_dir=output_dir,
                    sample_count=sample_count,
                    aspect_ratio=aspect_ratio,
                )

                # Format response for single prompt
                response_text = f"Successfully generated {len(result)} image(s):\n"
                for i, path in enumerate(result, 1):
                    response_text += f"{i}. {path}\n"

                return [TextContent(type="text", text=response_text.strip())]

            # Handle batch mode
            else:
                if prompts:
                    logger.info(f"Generating images for {len(prompts)} prompts in batch mode")
                    prompt_source = prompts
                else:
                    logger.info(
                        f"Generating images from {len(prompt_files)} prompt files in batch mode"
                    )
                    prompt_source = None

                result = imagen_client.generate_images(
                    prompt=prompt_source,
                    prompt_files=prompt_files,
                    model=model,
                    output_dir=output_dir or ".",
                    output_dirs=output_dirs,
                    sample_count=sample_count,
                    aspect_ratio=aspect_ratio,
                    max_workers=max_workers,
                )

                # Format response for batch mode
                num_prompts = len(prompts) if prompts else len(prompt_files)
                response_text = (
                    f"Batch image generation complete:\n"
                    f"Total prompts: {num_prompts}\n"
                    f"Successful: {result['successful']}\n"
                    f"Failed: {result['failed']}\n\n"
                )

                # Add details for each prompt
                for i, item in enumerate(result["results"], 1):
                    prompt_preview = item["prompt"][:50] + (
                        "..." if len(item["prompt"]) > 50 else ""
                    )
                    if item["error"]:
                        error_msg = item["error"]
                        response_text += f"{i}. ❌ '{prompt_preview}' - Error: {error_msg}\n"
                    else:
                        num_files = len(item["files"])
                        response_text += f"{i}. ✓ '{prompt_preview}' - {num_files} image(s)\n"
                        for file_path in item["files"]:
                            response_text += f"   → {file_path}\n"

                return [TextContent(type="text", text=response_text.strip())]

        elif name == "style-to-image":
            # Extract parameters for style-to-image
            prompt = arguments["prompt"]
            style_image_path = arguments["style_image_path"]
            style_description = arguments["style_description"]
            output_dir = arguments["output_dir"]
            sample_count = arguments.get("sample_count", 1)

            # Validate paths
            output_path = Path(output_dir)
            if not output_path.is_absolute():
                raise ValueError(f"output_dir must be an absolute path, got: {output_dir}")

            style_path = Path(style_image_path)
            if not style_path.is_absolute():
                raise ValueError(
                    f"style_image_path must be an absolute path, got: {style_image_path}"
                )

            logger.info(f"Generating styled images with prompt: {prompt[:100]}...")
            logger.info(f"Style reference: {style_image_path}")

            # Generate images with style
            file_paths = imagen_client.generate_images_with_style(
                prompt=prompt,
                style_image_path=style_image_path,
                style_description=style_description,
                output_dir=output_dir,
                sample_count=sample_count,
            )

            # Format response
            response_text = (
                f"Successfully generated {len(file_paths)} styled image(s) "
                f"following '{style_description}':\n"
            )
            for i, path in enumerate(file_paths, 1):
                response_text += f"{i}. {path}\n"

            return [TextContent(type="text", text=response_text.strip())]

        elif name == "remove-background":
            # Determine if single or batch processing
            input_path = arguments.get("input_path")
            input_paths = arguments.get("input_paths")
            output_dir = arguments.get("output_dir")
            overwrite = arguments.get("overwrite", True)
            max_workers = arguments.get("max_workers", 4)

            # Validate that either input_path or input_paths is provided
            if not input_path and not input_paths:
                raise ValueError("Either input_path or input_paths must be provided")

            if input_path and input_paths:
                raise ValueError("Cannot specify both input_path and input_paths")

            # Validate output_dir requirements
            if not overwrite and not output_dir:
                raise ValueError("output_dir is required when overwrite=False")

            # Handle single image processing
            if input_path:
                # Validate input path
                input_file_path = Path(input_path)
                if not input_file_path.is_absolute():
                    raise ValueError(f"input_path must be an absolute path, got: {input_path}")

                # Validate output_dir if provided
                if output_dir:
                    output_dir_path = Path(output_dir)
                    if not output_dir_path.is_absolute():
                        raise ValueError(f"output_dir must be an absolute path, got: {output_dir}")

                mode_desc = "overwrite mode" if overwrite else "preserve mode"
                logger.info(f"Removing background from single image ({mode_desc}): {input_path}")

                # Remove background
                result = ImagenClient.remove_background(
                    input_paths=input_path,
                    output_dir=output_dir,
                    overwrite=overwrite,
                )

                # Format response for single image
                if overwrite:
                    response_text = (
                        f"Successfully removed background (overwrote original):\n"
                        f"File: {result['output']}"
                    )
                else:
                    response_text = (
                        f"Successfully removed background:\n"
                        f"Input: {result['input']}\n"
                        f"Output: {result['output']}"
                    )

                return [TextContent(type="text", text=response_text)]

            # Handle batch processing
            else:
                # Validate input_paths
                if not isinstance(input_paths, list) or len(input_paths) == 0:
                    raise ValueError("input_paths must be a non-empty array")

                # Validate output_dir if provided
                if output_dir:
                    output_dir_path = Path(output_dir)
                    if not output_dir_path.is_absolute():
                        raise ValueError(f"output_dir must be an absolute path, got: {output_dir}")

                # Validate all input paths are absolute
                for path in input_paths:
                    if not Path(path).is_absolute():
                        raise ValueError(f"All input paths must be absolute, got: {path}")

                mode_desc = "overwrite mode" if overwrite else "preserve mode"
                logger.info(
                    f"Removing background from {len(input_paths)} images "
                    f"in batch mode ({mode_desc})"
                )

                # Remove backgrounds in parallel
                result = ImagenClient.remove_background(
                    input_paths=input_paths,
                    output_dir=output_dir,
                    overwrite=overwrite,
                    max_workers=max_workers,
                )

                # Format response for batch processing
                response_text = (
                    f"Batch background removal complete:\n"
                    f"Total: {len(input_paths)} images\n"
                    f"Successful: {result['successful']}\n"
                    f"Failed: {result['failed']}\n\n"
                )

                # Add details for each image
                for i, item in enumerate(result["results"], 1):
                    input_name = Path(item["input"]).name
                    if item["error"]:
                        error_msg = item["error"]
                        response_text += f"{i}. ❌ {input_name} - Error: {error_msg}\n"
                    else:
                        response_text += f"{i}. ✓ {input_name} → {item['output']}\n"

                return [TextContent(type="text", text=response_text.strip())]

        elif name == "autocrop":
            # Extract parameters for autocrop
            input_paths = arguments["input_paths"]
            output_dir = arguments.get("output_dir")
            padding = arguments.get("padding", 0)
            overwrite = arguments.get("overwrite", True)

            # Validate input paths
            if not isinstance(input_paths, list) or len(input_paths) == 0:
                raise ValueError("input_paths must be a non-empty list")

            for input_path in input_paths:
                input_file_path = Path(input_path)
                if not input_file_path.is_absolute():
                    raise ValueError(f"All input paths must be absolute, got: {input_path}")

            # Validate output directory if provided
            if output_dir:
                output_path = Path(output_dir)
                if not output_path.is_absolute():
                    raise ValueError(f"output_dir must be an absolute path, got: {output_dir}")
                output_path.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"Auto-cropping {len(input_paths)} image(s) with padding={padding}px, "
                f"overwrite={overwrite}"
            )

            # Process images in parallel using asyncio
            async def crop_single_image(input_path: str) -> tuple[str, str | None]:
                """Crop a single image and return (input_path, output_path or error)."""
                try:
                    # Determine output path based on overwrite and output_dir
                    if overwrite and not output_dir:
                        # When overwrite=True and no output_dir, overwrite the original file
                        output_path = input_path
                    elif output_dir:
                        # When output_dir is specified, save to output_dir
                        input_file = Path(input_path)
                        if overwrite:
                            # When overwrite=True, use the original filename
                            output_path = Path(output_dir) / input_file.name
                        else:
                            # When overwrite=False, use _cropped suffix
                            output_path = (
                                Path(output_dir) / f"{input_file.stem}_cropped{input_file.suffix}"
                            )
                    else:
                        # When overwrite=False and no output_dir, use _cropped suffix
                        output_path = None

                    # Run crop in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        ImagenClient.autocrop_image,
                        input_path,
                        str(output_path) if output_path else None,
                        padding,
                        overwrite,
                    )
                    return (input_path, result)
                except Exception as e:
                    logger.error(f"Error cropping {input_path}: {e}")
                    return (input_path, f"Error: {str(e)}")

            # Process all images in parallel
            results = await asyncio.gather(*[crop_single_image(path) for path in input_paths])

            # Separate successful and failed results
            successful = [(inp, out) for inp, out in results if not out.startswith("Error:")]
            failed = [(inp, out) for inp, out in results if out.startswith("Error:")]

            # Format response
            response_text = f"Processed {len(input_paths)} image(s):\n"
            response_text += f"Successfully cropped: {len(successful)}\n"

            if successful:
                response_text += "\nCropped images:\n"
                for i, (_, output_path) in enumerate(successful, 1):
                    response_text += f"{i}. {output_path}\n"

            if failed:
                response_text += f"\nFailed: {len(failed)}\n"
                for inp, error in failed:
                    response_text += f"- {Path(inp).name}: {error}\n"

            return [TextContent(type="text", text=response_text.strip())]

    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return [TextContent(type="text", text=error_msg)]


async def run_server():
    """Run the MCP server."""
    global imagen_client

    # Get Vertex AI configuration
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    # Try to get project from gcloud config if not set
    if not project:
        try:
            import subprocess

            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                project = result.stdout.strip()
                logger.info(f"Using project from gcloud config: {project}")
        except Exception as e:
            logger.warning(f"Could not get project from gcloud: {e}")

    if not project:
        logger.error(
            "No Google Cloud project configured. Please either:\n"
            "1. Set GOOGLE_CLOUD_PROJECT environment variable, or\n"
            "2. Configure gcloud default project: gcloud config set project PROJECT_ID"
        )
        sys.exit(1)

    logger.info(f"Using Vertex AI with project={project}, location={location}")
    imagen_client = ImagenClient(vertexai=True, project=project, location=location)

    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        logger.info("MCP Imagen server started")
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    """Main entry point."""
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
