import os
from typing import Any, Dict, List, Optional, Set, Tuple

import click
from dotenv import load_dotenv
from pymilvus import MilvusClient
from pytidb import TiDBClient
from pytidb.schema import Field, TableModel, VectorField
from tqdm import tqdm


def _load_env() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(dotenv_path=os.path.join(here, ".env"))


def _milvus_client() -> MilvusClient:
    _load_env()
    host = os.getenv("MILVUS_HOST")
    grpc_port = os.getenv("MILVUS_GRPC_PORT")
    if not host or not grpc_port:
        raise click.ClickException("Missing MILVUS_HOST / MILVUS_GRPC_PORT in env or .env")
    return MilvusClient(uri=f"http://{host}:{grpc_port}", token="root:Milvus")


def _tidb_client(enable_ssl: bool) -> TiDBClient:
    _load_env()
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    username = os.getenv("DB_USERNAME")
    password = os.getenv("DB_PASSWORD", "")
    database = os.getenv("DB_DATABASE")
    if not host or not port or not username or not database:
        raise click.ClickException(
            "Missing one of DB_HOST / DB_PORT / DB_USERNAME / DB_DATABASE in env or .env"
        )
    return TiDBClient.connect(
        host=host,
        port=int(port),
        username=username,
        password=password,
        database=database,
        enable_ssl=enable_ssl,
        ensure_db=True,
    )


def _make_chunk_model(table_name: str, embedding_dim: int):
    class Chunk(TableModel, table=True):
        __tablename__ = table_name

        id: int | None = Field(default=None, primary_key=True)
        chunk_text: str
        doc_id: str
        chunk_index: int
        source: str
        created_at: str
        embedding: Optional[Any] = VectorField(dimensions=embedding_dim, index=True)

    return Chunk


def _milvus_filter(start: int, end: Optional[int]) -> str:
    if end is None:
        return f"id >= {start}"
    return f"id >= {start} && id < {end}"


def _coerce_embedding(vec: Any) -> List[float]:
    if vec is None:
        return []
    if isinstance(vec, list):
        return [float(x) for x in vec]
    return [float(x) for x in list(vec)]


def _fetch_milvus_page(
    client: MilvusClient,
    collection: str,
    start: int,
    end: Optional[int],
    limit: int,
    offset: int,
) -> List[Dict[str, Any]]:
    return client.query(
        collection_name=collection,
        filter=_milvus_filter(start, end),
        output_fields=["id", "chunk_text", "doc_id", "chunk_index", "source", "created_at", "embedding"],
        limit=limit,
        offset=offset,
    )


def _existing_ids_in_tidb(table, ids: List[int]) -> Set[int]:
    if not ids:
        return set()
    ids_sql = ",".join(str(int(i)) for i in ids)
    rows = table.query(filters=f"id IN ({ids_sql})").to_list()
    return {int(r["id"]) for r in rows if "id" in r and r["id"] is not None}


def migrate_range(
    milvus_collection: str,
    tidb_table: str,
    start: int,
    end: Optional[int],
    page_size: int,
    enable_ssl: bool,
    dry_run: bool,
) -> Tuple[int, int, int]:
    milvus = _milvus_client()
    tidb = _tidb_client(enable_ssl=enable_ssl)

    if not milvus.has_collection(milvus_collection):
        raise click.ClickException(f"Milvus collection not found: {milvus_collection}")

    milvus.load_collection(milvus_collection)

    desc = milvus.describe_collection(milvus_collection)
    embedding_field = next((f for f in (desc.get("fields") or []) if f.get("name") == "embedding"), None)
    embedding_dim = int((embedding_field or {}).get("params", {}).get("dim", 1536))

    Chunk = _make_chunk_model(tidb_table, embedding_dim=embedding_dim)
    # `open_table()` only checks whether the model is mapped, not whether the physical table exists.
    if not tidb.has_table(tidb_table):
        table = tidb.create_table(schema=Chunk)
    else:
        table = tidb.open_table(tidb_table)
        if table is None:
            table = tidb.create_table(schema=Chunk)

    offset = 0
    scanned = 0
    skipped = 0
    inserted = 0

    progress = tqdm(desc="Migrating", unit="rows", total=None)
    try:
        while True:
            page = _fetch_milvus_page(
                client=milvus,
                collection=milvus_collection,
                start=start,
                end=end,
                limit=page_size,
                offset=offset,
            )
            if not page:
                break

            offset += len(page)
            scanned += len(page)
            progress.update(len(page))

            page_ids = [int(r["id"]) for r in page if "id" in r and r["id"] is not None]
            existing = _existing_ids_in_tidb(table, page_ids)

            to_insert: List[Dict[str, Any]] = []
            for r in page:
                rid = r.get("id")
                if rid is None:
                    continue
                rid_int = int(rid)
                if rid_int in existing:
                    skipped += 1
                    continue

                to_insert.append(
                    {
                        "id": rid_int,
                        "chunk_text": r.get("chunk_text", ""),
                        "doc_id": r.get("doc_id", ""),
                        "chunk_index": int(r.get("chunk_index", 0) or 0),
                        "source": r.get("source", ""),
                        "created_at": r.get("created_at", ""),
                        "embedding": _coerce_embedding(r.get("embedding")),
                    }
                )

            if not to_insert:
                continue

            if dry_run:
                inserted += len(to_insert)
                continue

            table.bulk_insert(to_insert)
            inserted += len(to_insert)
    finally:
        progress.close()

    return scanned, skipped, inserted


@click.command(help="Migrate chunk rows from Milvus to TiDB (re-entrant by primary key id).")
@click.option("--milvus-collection", default="chunk", show_default=True)
@click.option("--tidb-table", default="chunks", show_default=True)
@click.option("--start", type=int, default=0, show_default=True, help="Inclusive start id.")
@click.option("--end", type=int, default=None, help="Exclusive end id; if omitted, reads to the end.")
@click.option("--page-size", type=int, default=200, show_default=True)
@click.option("--ssl/--no-ssl", default=True, show_default=False, help="Enable SSL when connecting to TiDB.")
@click.option("--dry-run", is_flag=True, help="Do not write to TiDB; only simulate inserts.")
def main(
    milvus_collection: str,
    tidb_table: str,
    start: int,
    end: Optional[int],
    page_size: int,
    ssl: bool,
    dry_run: bool,
) -> None:
    if end is not None and end < start:
        raise click.ClickException("--end must be >= --start")

    scanned, skipped, inserted = migrate_range(
        milvus_collection=milvus_collection,
        tidb_table=tidb_table,
        start=start,
        end=end,
        page_size=page_size,
        enable_ssl=ssl,
        dry_run=dry_run,
    )
    click.echo(
        f"Done. scanned={scanned} skipped_existing={skipped} inserted={inserted} "
        f"(milvus_collection={milvus_collection}, tidb_table={tidb_table}, range=[{start},{end if end is not None else 'end'}))"
    )


if __name__ == "__main__":
    main()
