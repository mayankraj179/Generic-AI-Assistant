from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from app.config.knowledge_sources import ObjectStorageSourceConfig
from app.ingestion.pipeline import (
    DiscoveredSource,
    ParsedDocument,
    ParseError,
    SourceWithAccess,
    extract_text,
    is_supported_file,
    parsed_document,
)
from app.ingestion.sources.base import KnowledgeSource, SourceError, resolve_env_ref

# Bigger than any policy document should be; stops one huge object from
# being pulled into memory and parsed.
MAX_OBJECT_BYTES = 50 * 1024 * 1024


class ObjectStorageSource(KnowledgeSource):
    """An S3-compatible bucket + prefix (AWS S3, MinIO, ...). Lists and
    reads objects only. URIs are s3://<name>/<object key>."""

    source_type = "s3"

    def __init__(self, config: ObjectStorageSourceConfig, *, client: Any | None = None) -> None:
        super().__init__(config.name, config.access_labels)
        self.config = config
        self._client = client

    def _s3(self) -> Any:
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "s3",
                endpoint_url=self.config.endpoint_url,
                region_name=self.config.region,
                aws_access_key_id=resolve_env_ref(
                    self.config.access_key_id, field="access_key_id", source=self.name
                ),
                aws_secret_access_key=resolve_env_ref(
                    self.config.secret_access_key, field="secret_access_key", source=self.name
                ),
                config=Config(
                    connect_timeout=5,
                    read_timeout=30,
                    retries={"max_attempts": 3},
                    s3={"addressing_style": "path"},
                ),
            )
        return self._client

    def _list_keys(self) -> list[str]:
        paginator = self._s3().get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=self.config.prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if key.endswith("/") or not is_supported_file(key):
                    continue
                keys.append(key)
                if len(keys) > self.config.max_objects:
                    raise SourceError(
                        f"source '{self.name}': more than max_objects="
                        f"{self.config.max_objects} objects under "
                        f"s3://{self.config.bucket}/{self.config.prefix}"
                    )
        return sorted(keys)

    async def discover(self) -> list[DiscoveredSource]:
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            keys = await asyncio.to_thread(self._list_keys)
        except (BotoCoreError, ClientError) as exc:
            raise SourceError(
                f"source '{self.name}': could not list "
                f"s3://{self.config.bucket}/{self.config.prefix}: {exc}"
            ) from exc
        return [
            DiscoveredSource(uri=self.uri_for(key), source_type="object_storage") for key in keys
        ]

    def _download(self, key: str, target: Path) -> None:
        from botocore.exceptions import ClientError

        try:
            response = self._s3().get_object(Bucket=self.config.bucket, Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
                raise FileNotFoundError(f"object does not exist: {key}") from exc
            raise
        if response.get("ContentLength", 0) > MAX_OBJECT_BYTES:
            raise ParseError(f"object is larger than {MAX_OBJECT_BYTES} bytes: {key}")
        target.write_bytes(response["Body"].read())

    async def fetch(self, source: SourceWithAccess) -> ParsedDocument:
        from botocore.exceptions import BotoCoreError, ClientError

        key = self.locator_of(source.source.uri)
        name = PurePosixPath(key).name
        with tempfile.TemporaryDirectory() as tmp:
            # Same file name (and so suffix) as the object, for parser dispatch.
            path = Path(tmp) / name
            try:
                await asyncio.to_thread(self._download, key, path)
            except (BotoCoreError, ClientError) as exc:
                raise ParseError(f"could not read object {key}: {exc}") from exc
            raw_text = extract_text(path)
        return parsed_document(source, title=PurePosixPath(key).stem or name, raw_text=raw_text)
