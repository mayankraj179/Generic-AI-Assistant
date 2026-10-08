"""Seeds the local demo data behind configs/kb_demo_assistant.yaml.

Dev only. Run from backend/ with the compose stack up (postgres, minio):

    python scripts/seed_kb_demo.py

Idempotent. It sets up:
  - Postgres: database kb_demo_source with the office_directory table, and a
    login role that can only SELECT from it (read-only by default too). The
    role's name and password come from KB_DEMO_DB_URL.
  - MinIO: bucket kb-demo holding sample_sources/kb_demo/object_storage/,
    and a user that can only list and read that bucket (from
    KB_DEMO_S3_ACCESS_KEY / KB_DEMO_S3_SECRET_KEY). The user is created with
    the mc client in a throwaway container, since MinIO's admin API isn't S3.

Admin access uses DATABASE_URL and MINIO_ROOT_USER / MINIO_ROOT_PASSWORD
(defaulting to infra/docker-compose.yml's dev values).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

import boto3
from dotenv import dotenv_values
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

DEMO_DIR = Path(__file__).resolve().parent.parent / "sample_sources" / "kb_demo"
BUCKET = "kb-demo"
MC_IMAGE = (
    "cgr.dev/chainguard/minio-client@sha256:"
    "7a5387f27d1aa8fc11286acde9e024b7b1b77585e17c36378552eeafcccf2ba3"
)


def env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name) or dotenv_values(".env").get(name) or default
    if not value:
        sys.exit(f"{name} is not set (environment or backend/.env)")
    return value


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


async def seed_postgres() -> None:
    admin_url = make_url(env("DATABASE_URL"))
    reader_url = make_url(env("KB_DEMO_DB_URL"))
    database, reader, password = reader_url.database, reader_url.username, reader_url.password
    if not (database and reader and password):
        sys.exit("KB_DEMO_DB_URL must include a user, password and database")

    admin = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    async with admin.connect() as conn:
        exists = await conn.scalar(
            text("SELECT 1 FROM pg_database WHERE datname = :db"), {"db": database}
        )
        if not exists:
            await conn.execute(text(f"CREATE DATABASE {_ident(database)}"))
        role = await conn.scalar(text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": reader})
        verb = "ALTER" if role else "CREATE"
        await conn.execute(
            text(f"{verb} ROLE {_ident(reader)} LOGIN PASSWORD {_literal(password)}")
        )
        # Read-only even if someone later grants it more.
        await conn.execute(
            text(f"ALTER ROLE {_ident(reader)} SET default_transaction_read_only = on")
        )
    await admin.dispose()

    demo = create_async_engine(admin_url.set(database=database), isolation_level="AUTOCOMMIT")
    async with demo.connect() as conn:
        sql = (DEMO_DIR / "database" / "office_directory.sql").read_text(encoding="utf-8")
        for statement in sql.split(";"):
            body = "\n".join(
                line for line in statement.splitlines() if not line.strip().startswith("--")
            ).strip()
            if body:
                await conn.execute(text(body))
        await conn.execute(text(f"REVOKE ALL ON DATABASE {_ident(database)} FROM PUBLIC"))
        await conn.execute(
            text(f"GRANT CONNECT ON DATABASE {_ident(database)} TO {_ident(reader)}")
        )
        await conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {_ident(reader)}"))
        await conn.execute(text(f"GRANT SELECT ON office_directory TO {_ident(reader)}"))
        rows = await conn.scalar(text("SELECT count(*) FROM office_directory"))
    await demo.dispose()
    print(
        f"postgres: database {database}, table office_directory ({rows} rows), "
        f"read-only role {reader}"
    )


def seed_minio(endpoint: str, network: str) -> None:
    root_user = env("MINIO_ROOT_USER", "minio_admin")
    root_password = env("MINIO_ROOT_PASSWORD", "minio_dev_password")
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
        aws_access_key_id=root_user,
        aws_secret_access_key=root_password,
    )
    if BUCKET not in {b["Name"] for b in s3.list_buckets().get("Buckets", [])}:
        s3.create_bucket(Bucket=BUCKET)
    root = DEMO_DIR / "object_storage"
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        key = path.relative_to(root).as_posix()
        s3.upload_file(str(path), BUCKET, key)
        print(f"minio: uploaded s3://{BUCKET}/{key}")

    reader, reader_secret = env("KB_DEMO_S3_ACCESS_KEY"), env("KB_DEMO_S3_SECRET_KEY")

    def mc(*args: str, tolerate: str | None = None) -> None:
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                network,
                "-e",
                f"MC_HOST_kb=http://{root_user}:{root_password}@minio:9000",
                "-v",
                f"{DEMO_DIR}:/kb:ro",
                MC_IMAGE,
                *args,
            ],
            capture_output=True,
            text=True,
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode != 0 and not (tolerate and tolerate in output):
            sys.exit(f"mc {' '.join(args[:3])} failed: {output}")

    mc("admin", "policy", "create", "kb", "kb-demo-read", "/kb/minio_read_policy.json")
    mc("admin", "user", "add", "kb", reader, reader_secret)
    mc(
        "admin",
        "policy",
        "attach",
        "kb",
        "kb-demo-read",
        "--user",
        reader,
        tolerate="already attached",
    )
    print(f"minio: user {reader} can only list and read bucket {BUCKET}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--minio-endpoint", default="http://localhost:9000")
    parser.add_argument(
        "--docker-network",
        default="infra_default",
        help="compose network the minio container is on",
    )
    args = parser.parse_args()
    asyncio.run(seed_postgres())
    seed_minio(args.minio_endpoint, args.docker_network)


if __name__ == "__main__":
    main()
