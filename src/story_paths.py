from pathlib import Path
from typing import Optional

from src import config


def get_story_name(source: Path) -> str:
    source = Path(source).resolve()
    if source.is_file():
        source = source.parent
    if source.name in {"media", "videos", "parts"}:
        source = source.parent
    if source.name:
        return source.name
    return "story"


def get_story_output_root(source: Optional[Path] = None) -> Path:
    if source is None:
        return config.OUTPUT_DIR
    source = Path(source).resolve()

    input_root = config.INPUT_DIR.resolve()
    output_root = config.OUTPUT_DIR.resolve()
    try:
        source.relative_to(input_root)
        return config.OUTPUT_DIR / get_story_name(source)
    except ValueError:
        pass

    try:
        source.relative_to(output_root)
        if source.is_file():
            source = source.parent
        if source.name in {"media", "videos", "parts"}:
            source = source.parent
        relative = source.relative_to(output_root)
        if len(relative.parts) >= 1:
            return output_root / relative.parts[0]
    except ValueError:
        pass

    if source.is_file():
        source = source.parent
    if source.name in {"media", "videos", "parts"}:
        source = source.parent
    return source
