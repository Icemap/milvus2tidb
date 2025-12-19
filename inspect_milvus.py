import json
import os
from typing import Any, Dict, List, Optional

import click
from dotenv import load_dotenv
from pymilvus import MilvusClient


def _load_env() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(dotenv_path=os.path.join(here, ".env"))


def _client() -> MilvusClient:
    _load_env()
    host = os.getenv("MILVUS_HOST")
    grpc_port = os.getenv("MILVUS_GRPC_PORT")
    if not host or not grpc_port:
        raise click.ClickException("Missing MILVUS_HOST / MILVUS_GRPC_PORT in env or .env")
    return MilvusClient(uri=f"http://{host}:{grpc_port}", token="root:Milvus")


def _format_fields(fields: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for f in fields:
        name = f.get("name")
        dtype = f.get("type")
        is_primary = f.get("is_primary_key", f.get("is_primary", False))
        auto_id = f.get("auto_id", None)
        params = f.get("params") or {}
        extras = []
        if is_primary:
            extras.append("primary")
        if auto_id is True:
            extras.append("auto_id")
        if params:
            extras.append(f"params={params}")
        suffix = f" ({', '.join(extras)})" if extras else ""
        lines.append(f"- {name}: {dtype}{suffix}")
    return "\n".join(lines) if lines else "(no fields)"


def _query_sample(
    client: MilvusClient,
    collection: str,
    limit: int,
    include_embedding: bool,
) -> List[Dict[str, Any]]:
    client.load_collection(collection)
    output_fields = ["id", "chunk_text", "doc_id", "chunk_index", "source", "created_at"]
    if include_embedding:
        output_fields.append("embedding")
    rows = client.query(
        collection_name=collection,
        filter="id >= 0",
        output_fields=output_fields,
        limit=limit,
    )
    # pymilvus may return QueryResult-like objects; normalize to plain dicts
    normalized: List[Dict[str, Any]] = []
    for r in rows:
        if isinstance(r, dict):
            normalized.append(r)
        else:
            try:
                normalized.append(dict(r))
            except Exception:
                normalized.append({"_raw": str(r)})
    if not include_embedding:
        return normalized
    for r in normalized:
        emb = r.get("embedding")
        if isinstance(emb, list):
            r["embedding_dim"] = len(emb)
            r["embedding_preview"] = emb[:5]
            r.pop("embedding", None)
    return normalized


@click.command(help="Inspect Milvus collection schema and row count, and optionally print sample rows.")
@click.option("--collection", default="chunk", show_default=True)
@click.option("--flush/--no-flush", default=True, show_default=True, help="Flush collection before counting.")
@click.option("--sample", type=int, default=5, show_default=True, help="Number of sample rows to print (0 to disable).")
@click.option("--include-embedding", is_flag=True, help="Show embedding dim + preview in sample output.")
@click.option("--json-output", is_flag=True, help="Print as JSON (machine-readable).")
def main(
    collection: str,
    flush: bool,
    sample: int,
    include_embedding: bool,
    json_output: bool,
) -> None:
    client = _client()
    if not client.has_collection(collection):
        raise click.ClickException(f"Collection not found: {collection}")

    if flush:
        client.flush(collection)

    desc = client.describe_collection(collection)
    stats = client.get_collection_stats(collection)
    row_count = stats.get("row_count")

    sample_rows: Optional[List[Dict[str, Any]]] = None
    if sample > 0:
        sample_rows = _query_sample(client, collection, limit=sample, include_embedding=include_embedding)

    if json_output:
        payload = {
            "collection": collection,
            "describe_collection": desc,
            "collection_stats": stats,
            "sample_rows": sample_rows,
        }
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    click.echo(f"Collection: {collection}")
    click.echo(f"Row count: {row_count}")
    click.echo(f"Enable dynamic field: {desc.get('enable_dynamic_field')}")
    click.echo("Fields:")
    click.echo(_format_fields(desc.get("fields") or []))

    if sample_rows is not None:
        click.echo("")
        click.echo(f"Sample rows (n={len(sample_rows)}):")
        click.echo(json.dumps(sample_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
