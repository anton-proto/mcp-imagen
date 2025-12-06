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
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name != "text-to-image":
        raise ValueError(f"Unknown tool: {name}")

    if not imagen_client:
        raise RuntimeError("Imagen client not initialized")

    try:
        # Extract parameters
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

    except Exception as e:
        error_msg = f"Error generating images: {str(e)}"
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
    imagen_client = ImagenClient(project=project, location=location)

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
