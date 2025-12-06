# MCP Imagen Server

An MCP (Model Context Protocol) server for Google Imagen API, enabling text-to-image generation using Google's state-of-the-art Imagen models.

## Features

- **Text-to-Image Generation**: Generate high-quality images from text prompts using Imagen 4.0 models
- **Style Transfer**: Generate images following the style of a reference image using Imagen 3 Customization
- **Background Removal**: Remove backgrounds from images using rembg AI model
- **Auto-Crop**: Automatically crop images to remove transparent or empty borders with batch processing support
- **Multiple Models**: Support for three Imagen variants:
  - `imagen-4.0-generate-001` (default) - Standard quality and speed
  - `imagen-4.0-fast-generate-001` - Faster generation
  - `imagen-4.0-ultra-generate-001` - Highest quality (single image only)
- **Flexible Configuration**:
  - Customizable aspect ratios (1:1, 3:4, 4:3, 9:16, 16:9)
  - Batch generation (1-4 images per request)
  - PNG output format with transparency support
- **Authentication Options**:
  - Google Cloud Default Application Credentials
  - Vertex AI or Gemini API

## Prerequisites

- Python 3.11 or later
- [uv](https://github.com/astral-sh/uv) for package management
- Google Cloud credentials (see [Authentication](#authentication))

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/anton-proto/mcp-imagen.git
cd mcp-imagen/mcp-imagen-server
```

### 2. Install with uv

```bash
# Install dependencies
uv sync

# Or install in development mode
uv sync --all-extras
```

### 3. Set Up Authentication

#### Option A: Using Gemini API (Recommended for quick start)

1. Get an API key from [Google AI Studio](https://aistudio.google.com/apikey)
2. Set the API key:
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

#### Option B: Using Vertex AI (Recommended for production)

1. Install Google Cloud SDK:
   ```bash
   # For Debian/Ubuntu
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   ```

2. Authenticate with Google Cloud:
   ```bash
   gcloud auth application-default login
   ```

3. Set your project:
   ```bash
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   export USE_VERTEXAI="true"
   # Optional: specify location (default: us-central1)
   export GOOGLE_CLOUD_LOCATION="us-central1"
   ```

## Usage

### Running the Server

```bash
uv run mcp-imagen-server
```

The server will start and listen for MCP requests via stdio.

### Integration with Claude Desktop

Add this configuration to your Claude Desktop config file:

**Location**:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

**Configuration**:

```json
{
  "mcpServers": {
    "imagen": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/mcp-imagen-server",
        "run",
        "mcp-imagen-server"
      ],
      "env": {
        "GOOGLE_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

**For Vertex AI**:

```json
{
  "mcpServers": {
    "imagen": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/mcp-imagen-server",
        "run",
        "mcp-imagen-server"
      ],
      "env": {
        "USE_VERTEXAI": "true",
        "GOOGLE_CLOUD_PROJECT": "your-project-id",
        "GOOGLE_CLOUD_LOCATION": "us-central1"
      }
    }
  }
}
```

### Integration with Other MCP Clients

The server implements the standard MCP protocol and can be used with any MCP-compatible client.

## MCP Tools

### text-to-image

Generates images from text prompts using Google Imagen API.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Text description of the image to generate |
| `output_dir` | string | Yes | - | Absolute path to directory where images should be saved |
| `model` | string | No | `imagen-4.0-generate-001` | Imagen model to use (see [Models](#models)) |
| `sample_count` | integer | No | 1 | Number of images to generate (1-4, must be 1 for ultra model) |
| `aspect_ratio` | string | No | `1:1` | Aspect ratio of generated images (1:1, 3:4, 4:3, 9:16, 16:9) |

### Models

- **imagen-4.0-generate-001**: Standard model with balanced quality and speed
- **imagen-4.0-fast-generate-001**: Faster generation with good quality
- **imagen-4.0-ultra-generate-001**: Highest quality, single image only (sample_count must be 1)

### Response

Returns a text response with paths to generated PNG files:

```
Successfully generated 2 image(s):
1. /path/to/output/A_serene_mountain_landscape_at_sunset_1.png
2. /path/to/output/A_serene_mountain_landscape_at_sunset_2.png
```

### Example Usage

In Claude Desktop or other MCP client:

```
Generate an image of "A serene mountain landscape at sunset with a lake reflecting the sky" and save it to /tmp/images/
```

### style-to-image

Generates images following the style of a reference image using Imagen 3 Customization.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | string | Yes | Text description of the image content to generate |
| `style_image_path` | string | Yes | Absolute path to the style reference image file |
| `style_description` | string | Yes | Description of the style (e.g., "watercolor painting style", "neon sign style", "mosaic style") |
| `output_dir` | string | Yes | Absolute path to directory where images should be saved |
| `sample_count` | integer | No | Number of images to generate (1-4). Default: 1 |

#### Response

Returns a text response with paths to generated styled PNG files.

#### Example Usage

```
Generate an image of "A cat sitting on a windowsill" in the style of the image at /path/to/watercolor.png (watercolor painting style) and save it to /tmp/images/
```

### remove-background

Removes the background from an image using the rembg AI model, producing a PNG with transparent background.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input_path` | string | Yes | Absolute path to the input image file |
| `output_path` | string | No | Absolute path to save the output image. If not provided, will save with 'nobg_' prefix in same directory |

#### Response

Returns a text response with the path to the output image with transparent background:

```
Successfully removed background from image:
Output: /path/to/output/nobg_image.png
```

#### Example Usage

```
Remove the background from /home/user/images/photo.png
```

Or with custom output path:

```
Remove the background from /home/user/images/photo.png and save it to /home/user/outputs/transparent.png
```

### autocrop

Automatically crop images to remove transparent or empty borders. Supports both single image and batch processing with parallel execution.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input_paths` | array of strings | Yes | List of absolute paths to input image files to crop |
| `output_dir` | string | No | Absolute path to output directory. If not provided, cropped images will be saved in the same directory as input files with '_cropped' suffix |
| `padding` | integer | No | Number of pixels to add as padding around cropped content. Default: 0 |
| `overwrite` | boolean | No | Whether to overwrite existing output files. Default: True. If False and output file exists, the operation will fail with an error |

#### Response

Returns a text response with processing summary and paths to cropped images:

```
Processed 3 image(s):
Successfully cropped: 3

Cropped images:
1. /output/dir/image1_cropped.png
2. /output/dir/image2_cropped.png
3. /output/dir/image3_cropped.png
```

If any images fail to process, they will be listed separately:

```
Processed 3 image(s):
Successfully cropped: 2

Cropped images:
1. /output/dir/image1_cropped.png
2. /output/dir/image2_cropped.png

Failed: 1
- image3.png: Error: Image appears to be completely transparent or empty - cannot autocrop
```

#### Features

- **Parallel Processing**: Multiple images are processed concurrently for better performance
- **Batch Support**: Process multiple images in a single call
- **Flexible Output**: Save to a specific directory or use default location
- **Padding Control**: Add padding around cropped content if needed
- **Transparency Aware**: Automatically detects and crops around non-transparent pixels

#### Example Usage

Single image:
```
Autocrop the image at /home/user/images/logo.png
```

Multiple images with output directory:
```
Autocrop these images: ["/home/user/images/logo1.png", "/home/user/images/logo2.png", "/home/user/images/logo3.png"] and save to /home/user/cropped/
```

With padding:
```
Autocrop /home/user/images/logo.png with 10 pixels of padding and save to /home/user/output/
```

With overwrite disabled (prevent overwriting):
```
Autocrop /home/user/images/logo.png and save to /home/user/output/ with overwrite disabled
```

Note: By default, the tool will overwrite existing output files. Set `overwrite=False` to prevent accidental overwrites and raise an error if the output file already exists.

## Development

### Project Structure

```
mcp-imagen-server/
├── src/
│   └── mcp_imagen_server/
│       ├── __init__.py          # Package initialization
│       ├── imagen_client.py     # Imagen API client
│       └── server.py            # MCP server implementation
├── pyproject.toml               # Project configuration
├── README.md                    # This file
└── .python-version              # Python version
```

### Code Quality

The project uses [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check code
uv run ruff check src/

# Format code
uv run ruff format src/

# Auto-fix issues
uv run ruff check --fix src/
```

### Running Tests

```bash
# Run tests (when implemented)
uv run pytest
```

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GOOGLE_API_KEY` | Google AI API key for Gemini API | For Gemini API | - |
| `USE_VERTEXAI` | Set to "true" to use Vertex AI | No | false |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID | For Vertex AI | - |
| `GOOGLE_CLOUD_LOCATION` | GCP region | No | us-central1 |

## Troubleshooting

### Authentication Issues

**Problem**: `DefaultCredentialsError` or authentication failures

**Solution**:
- For Gemini API: Ensure `GOOGLE_API_KEY` is set
- For Vertex AI: Run `gcloud auth application-default login`
- Verify your project has the Vertex AI API enabled

### Permission Denied

**Problem**: Cannot write to output directory

**Solution**: Ensure the specified `output_dir` exists and is writable, or the server has permissions to create it

### Model Not Available

**Problem**: Model not found or access denied

**Solution**:
- Verify your Google Cloud project has access to Imagen models
- Check that you're using a supported model name
- For ultra model, ensure `sample_count=1`

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

- [Google Imagen Documentation](https://ai.google.dev/gemini-api/docs/imagen)
- [Vertex AI Imagen API](https://cloud.google.com/vertex-ai/generative-ai/docs/models/imagen/4-0-generate)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Google Gen AI Python SDK](https://github.com/googleapis/python-genai)
