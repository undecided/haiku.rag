from pathlib import Path

from watchfiles import Change, DefaultFilter, awatch

from haiku.rag.logging import get_logger
from haiku.rag.reader import FileReader
from haiku.rag.store.models.document import Document

logger = get_logger()


class FileFilter(DefaultFilter):
    def __init__(self, *, ignore_paths: list[Path] | None = None) -> None:
        self.extensions = tuple(FileReader.extensions)
        super().__init__(ignore_paths=ignore_paths)

    def __call__(self, change: "Change", path: str) -> bool:
        return path.endswith(self.extensions) and super().__call__(change, path)


class FileWatcher:
    def __init__(self, paths: list[Path]):
        self.paths = paths

    async def observe(self):
        logger.info(f"Watching files in {self.paths}")
        filter = FileFilter()
        await self.refresh()

        async for changes in awatch(*self.paths, watch_filter=filter):
            await self.handler(changes)

    async def handler(self, changes: set[tuple[Change, str]]):
        for change, path in changes:
            if change == Change.added or change == Change.modified:
                await self._upsert_document(Path(path))
            elif change == Change.deleted:
                await self._delete_document(Path(path))

    async def refresh(self):
        for path in self.paths:
            for f in Path(path).rglob("**/*"):
                if f.is_file() and f.suffix in FileReader.extensions:
                    await self._upsert_document(f)

    async def _delete_document(self, file: Path):
        logger.info(f"Deleting document from {file}")
        pass

    async def _upsert_document(self, file: Path) -> Document | None:
        logger.info(f"Updating document from {file}")
        pass
