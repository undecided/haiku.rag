from pathlib import Path
from typing import ClassVar

from markitdown import MarkItDown


class FileReader:
    extensions: ClassVar[list[str]] = [
        ".astro",
        ".c",
        ".cpp",
        ".css",
        ".csv",
        ".docx",
        ".go",
        ".h",
        ".hpp",
        ".html",
        ".java",
        ".js",
        ".json",
        ".kt",
        ".md",
        ".mdx",
        ".mjs",
        ".mp3",
        ".pdf",
        ".php",
        ".pptx",
        ".py",
        ".rb",
        ".rs",
        ".svelte",
        ".swift",
        ".ts",
        ".tsx",
        ".txt",
        ".vue",
        ".wav",
        ".xml",
        ".xlsx",
        ".yaml",
        ".yml",
    ]

    @staticmethod
    def parse_file(path: Path) -> str:
        try:
            reader = MarkItDown()
            return reader.convert(path).text_content
        except Exception:
            raise ValueError(f"Failed to parse file: {path}")
