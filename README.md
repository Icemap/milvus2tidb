# milvus2tidb

Migrate chunk data from Milvus to TiDB, with helper scripts to insert, inspect, and migrate data.

## Environment Variables

This project loads environment variables from `.env` in the repo root by default.

### Milvus

```env
MILVUS_HOST=127.0.0.1
MILVUS_GRPC_PORT=19530
```

### TiDB

```env
DB_HOST=127.0.0.1
DB_PORT=4000
DB_USERNAME=root
DB_PASSWORD=
DB_DATABASE=test
```

## Scripts

### 1. Insert mock chunks into Milvus

File: `insert_milvus.py`

```bash
python insert_milvus.py --docs 3 --chunks-per-doc 5 --recreate
```

Options:

- `--recreate`: drop and recreate the collection first
- `--batch-size`: batch size for inserts

### 2. Inspect Milvus schema and row count

File: `inspect_milvus.py`

```bash
python inspect_milvus.py --collection chunk --sample 5
python inspect_milvus.py --collection chunk --sample 2 --include-embedding
python inspect_milvus.py --collection chunk --json-output
```

### 3. Migrate: Milvus -> TiDB (re-entrant + id range)

File: `migrate_milvus_to_tidb.py`

Features:

- Range: `[start, end)` (inclusive start, exclusive end). If `--end` is omitted, it reads to the end.
- Re-entrant: if the row already exists in TiDB (by primary key `id`), it will be skipped; otherwise it will be inserted.

Examples:

```bash
# Migrate everything (from 0 to the end)
python migrate_milvus_to_tidb.py --start 0

# Migrate range [1, 16)
python migrate_milvus_to_tidb.py --start 1 --end 16

# Dry-run (no writes)
python migrate_milvus_to_tidb.py --start 0 --dry-run
```

Options:

- `--milvus-collection`: default `chunk`
- `--tidb-table`: default `chunks` (auto-created if missing)
- `--page-size`: Milvus query page size
- `--ssl/--no-ssl`: enable SSL for TiDB (depending on your cluster settings)
