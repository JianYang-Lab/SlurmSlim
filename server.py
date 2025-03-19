from pathlib import Path

import mistune
from mcp.server.fastmcp import FastMCP
from pygments import highlight
from pygments.lexers import get_lexer_for_filename, guess_lexer


class ContentBuilder:
    def __init__(self, content, file_type):
        self.content = content

    @classmethod
    def from_file(cls, file_path: str | Path):
        language, code = _preprocess_script(file_path)

        return cls(code, language)

    def build(self):
        return mistune.markdown(self.content)


def _wrap_content(content: str, file_type: str):
    return f"```{{{file_type}}}\n{content}\n```"


def _preprocess_script(file_path: str | Path):
    if isinstance(file_path, str):
        file_path = Path(file_path)

    code = None
    if file_path and file_path.exists():
        try:
            lexer = get_lexer_for_filename(file_path)
            with open(file_path, "r") as f:
                code = f.read()
        except Exception as e:
            print(f"Error detecting language from filename: {e}")
            # Fallback to content-based detection
            with open(file_path, "r") as f:
                code = f.read()
            lexer = guess_lexer(code)
    elif code is not None:
        # Try to guess the lexer based on content
        lexer = guess_lexer(code)
    else:
        raise ValueError("Either file_path or code must be provided")
    language = lexer.aliases[0] if lexer.aliases else "text"
    code = _wrap_content(code, language)
    return language, code

from pathlib import Path

# Initialize FastMCP server
mcp = FastMCP("optslurm")


@mcp.tool()
async def get_script_contents(file_path: str | Path) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    code_blocks = ContentBuilder.from_file(file_path).build()
    return code_blocks  # type: ignore


@mcp.tool()
async def get_file_size(file_path: str | Path) -> str:
    """Get weather forecast for a location.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if file_path.exists():
        return f"File size: {file_path.stat().st_size / 1024 / 1024 } MB"
    else:
        return "File not found"


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')