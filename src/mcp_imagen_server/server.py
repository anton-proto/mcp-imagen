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
                "Generate images from text prompts using Google Imagen API. "
                "Returns paths to generated PNG files."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image to generate",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Absolute path to directory where images should be saved",
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
                            "Number of images to generate (1-4). Must be 1 for ultra model. "
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
                },
                "required": ["prompt", "output_dir"],
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
                "Remove background from an image using rembg AI model. "
                "Returns path to the output image with transparent background."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Absolute path to the input image file",
                    },
                    "output_path": {
                        "type": "string",
                        "description": (
                            "Absolute path to save the output image (optional). "
                            "If not provided, will save with 'nobg_' prefix in same directory."
                        ),
                    },
                },
                "required": ["input_path"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name not in ["text-to-image", "style-to-image", "remove-background"]:
        raise ValueError(f"Unknown tool: {name}")

    if not imagen_client and name != "remove-background":
        raise RuntimeError("Imagen client not initialized")

    try:
        if name == "text-to-image":
            # Extract parameters for text-to-image
            prompt = arguments["prompt"]
            output_dir = arguments["output_dir"]
            model: ImagenModel = arguments.get("model", "imagen-4.0-generate-001")
            sample_count = arguments.get("sample_count", 1)
            aspect_ratio: AspectRatio = arguments.get("aspect_ratio", "1:1")

            # Validate output directory
            output_path = Path(output_dir)
            if not output_path.is_absolute():
                raise ValueError(f"output_dir must be an absolute path, got: {output_dir}")

            logger.info(f"Generating images with prompt: {prompt[:100]}...")

            # Generate images
            file_paths = imagen_client.generate_images(
                prompt=prompt,
                model=model,
                output_dir=output_dir,
                sample_count=sample_count,
                aspect_ratio=aspect_ratio,
            )

            # Format response
            response_text = f"Successfully generated {len(file_paths)} image(s):\n"
            for i, path in enumerate(file_paths, 1):
                response_text += f"{i}. {path}\n"

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
            # Extract parameters for remove-background
            input_path = arguments["input_path"]
            output_path = arguments.get("output_path")

            # Validate input path
            input_file_path = Path(input_path)
            if not input_file_path.is_absolute():
                raise ValueError(f"input_path must be an absolute path, got: {input_path}")

            # Validate output path if provided
            if output_path:
                output_file_path = Path(output_path)
                if not output_file_path.is_absolute():
                    raise ValueError(f"output_path must be an absolute path, got: {output_path}")

            logger.info(f"Removing background from: {input_path}")

            # Remove background
            output_file = ImagenClient.remove_background(
                input_path=input_path,
                output_path=output_path,
            )

            # Format response
            response_text = f"Successfully removed background from image:\nOutput: {output_file}"

            return [TextContent(type="text", text=response_text)]

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
