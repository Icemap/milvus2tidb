from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType
import os

from typing import List, Dict, Any
import hashlib
import random
from datetime import datetime, timezone

import click
from tqdm import tqdm

load_dotenv()

embedding_dimensions = 1536

milvus_client = MilvusClient(
    uri=f"http://{os.getenv('MILVUS_HOST')}:{os.getenv('MILVUS_GRPC_PORT')}",
    token="root:Milvus"
)
milvus_collection_name = "chunk"

def fake_embedding(text: str, dim: int) -> List[float]:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]

def iter_mock_chunks(
    start_id: int,
    num_docs: int,
    chunks_per_doc: int,
) -> Any:
    now_iso = datetime.now(timezone.utc).isoformat()
    next_id = start_id
    for doc_i in range(num_docs):
        doc_id = f"doc_{doc_i:04d}"
        source = f"mock://{doc_id}"
        for chunk_i in range(chunks_per_doc):
            chunk_text = (
                f"[{doc_id}] chunk={chunk_i} "
                f"This is a demo text to insert into Milvus."
                f"source={source} created_at={now_iso}"
            )
            yield {
                "id": next_id,
                "chunk_text": chunk_text,
                "embedding": fake_embedding(chunk_text, embedding_dimensions),
                "doc_id": doc_id,
                "chunk_index": chunk_i,
                "source": source,
                "created_at": now_iso,
            }
            next_id += 1

# Define the employee id mapping schema in Milvus
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="chunk_text", datatype=DataType.VARCHAR, max_length=65535)
schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=embedding_dimensions)
schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=512)
schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=2048)
schema.add_field(field_name="created_at", datatype=DataType.VARCHAR, max_length=64)

index_params = milvus_client.prepare_index_params()
index_params.add_index(
    field_name="embedding",
    index_type="AUTOINDEX"
)


def ensure_collection(collection_name: str, recreate: bool) -> None:
    if milvus_client.has_collection(collection_name):
        if recreate:
            milvus_client.drop_collection(collection_name)
        else:
            return
    milvus_client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )


def insert_mock_chunks(
    collection_name: str,
    start_id: int = 1,
    num_docs: int = 3,
    chunks_per_doc: int = 5,
    batch_size: int = 128,
    recreate: bool = False,
) -> int:
    ensure_collection(collection_name, recreate=recreate)
    total = num_docs * chunks_per_doc
    inserted_total = 0
    batch: List[Dict[str, Any]] = []

    for row in tqdm(
        iter_mock_chunks(
            start_id=start_id,
            num_docs=num_docs,
            chunks_per_doc=chunks_per_doc,
        ),
        total=total,
        desc="Inserting chunks",
    ):
        batch.append(row)
        if len(batch) < batch_size:
            continue
        result = milvus_client.insert(collection_name=collection_name, data=batch)
        inserted_total += int(result.get("insert_count", len(batch))) if isinstance(result, dict) else len(batch)
        batch = []

    if batch:
        result = milvus_client.insert(collection_name=collection_name, data=batch)
        inserted_total += int(result.get("insert_count", len(batch))) if isinstance(result, dict) else len(batch)

    return inserted_total


@click.command(help="Insert mock chunk data into Milvus.")
@click.option("--collection", default=milvus_collection_name, show_default=True)
@click.option("--start-id", type=int, default=1, show_default=True)
@click.option("--docs", type=int, default=3, show_default=True)
@click.option("--chunks-per-doc", type=int, default=5, show_default=True)
@click.option("--batch-size", type=int, default=128, show_default=True)
@click.option("--recreate", is_flag=True, help="Drop and recreate collection first.")
def main(collection: str, start_id: int, docs: int, chunks_per_doc: int, batch_size: int, recreate: bool) -> None:
    inserted = insert_mock_chunks(
        collection_name=collection,
        start_id=start_id,
        num_docs=docs,
        chunks_per_doc=chunks_per_doc,
        batch_size=batch_size,
        recreate=recreate,
    )
    click.echo(f"Inserted {inserted} mock chunks into collection '{collection}'.")


if __name__ == "__main__":
    main()
