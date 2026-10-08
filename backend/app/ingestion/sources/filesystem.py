from __future__ import annotations

from pathlib import Path

from app.config.knowledge_sources import FilesystemSourceConfig
from app.ingestion.pipeline import (
    DiscoveredSource,
    ParsedDocument,
    SourceWithAccess,
    discover,
    extract_text,
    parsed_document,
)
from app.ingestion.sources.base import KnowledgeSource, SourceError


class FilesystemSource(KnowledgeSource):
    """Local or uploaded documents: the existing discover() and parsers,
    behind the source interface. URIs are relative to the configured root
    (fs://<name>/policies/leave.md), so moving the root doesn't re-embed."""

    source_type = "fs"

    def __init__(self, config: FilesystemSourceConfig, *, base_dir: Path) -> None:
        super().__init__(config.name, config.access_labels)
        root = Path(config.path)
        self.root = (root if root.is_absolute() else base_dir / root).resolve()

    def _relative(self, path: Path) -> str:
        if self.root.is_file():
            return path.name
        return path.resolve().relative_to(self.root).as_posix()

    async def discover(self) -> list[DiscoveredSource]:
        if not self.root.exists():
            raise SourceError(f"source '{self.name}': path does not exist: {self.root}")
        return [
            DiscoveredSource(
                uri=self.uri_for(self._relative(Path(found.uri))), source_type="filesystem"
            )
            for found in discover(str(self.root))
        ]

    def _path_for(self, uri: str) -> Path:
        locator = self.locator_of(uri)
        if self.root.is_file():
            path = self.root if locator == self.root.name else None
        else:
            path = (self.root / locator).resolve()
            # A locator can't climb out of the configured root.
            if not path.is_relative_to(self.root):
                path = None
        if path is None or not path.is_file():
            raise FileNotFoundError(f"source file does not exist: {uri}")
        return path

    async def fetch(self, source: SourceWithAccess) -> ParsedDocument:
        path = self._path_for(source.source.uri)
        return parsed_document(source, title=path.stem or path.name, raw_text=extract_text(path))
