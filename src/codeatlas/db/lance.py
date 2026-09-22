"""LanceDB 向量表管理(PLAN §7)。

单表 chunks,行 = {id(uuid), vector(float32[dim]), chunk_id, repo_id, kind};
引用归 SQL 管(vector_refs 记 lance_id),精确删除按 chunk_id 过滤。
首次索引完成后行数足够时建 IVF_PQ(cosine)ANN 索引。
"""

from __future__ import annotations

import uuid

from codeatlas import config
from codeatlas.config import Settings
from codeatlas.providers.embedding import unpack_vector

# ANN 索引需要的最小行数(IVF 聚类中心要求,不足时跳过建索引、用暴力扫描)
MIN_ROWS_FOR_INDEX = 2048


class LanceStore:
    def __init__(self, settings: Settings):
        import lancedb
        import pyarrow as pa

        self.s = settings
        self._pa = pa
        self.db = lancedb.connect(str(config.LANCEDB_DIR))
        self._table = None

    @property
    def table(self):
        if self._table is None:
            import lancedb

            try:
                self._table = self.db.open_table("chunks")
            except Exception:
                pa = self._pa
                schema = pa.schema(
                    [
                        pa.field("id", pa.string()),
                        pa.field("vector", pa.list_(pa.float32(), self.s.embed_dim)),
                        pa.field("chunk_id", pa.int64()),
                        pa.field("repo_id", pa.int64()),
                        pa.field("kind", pa.string()),
                    ]
                )
                self._table = self.db.create_table(
                    "chunks", schema=schema, mode="create"
                )
        return self._table

    def add_vectors(
        self, rows: list[tuple[int, int, str, bytes]]
    ) -> list[tuple[str, int]]:
        """rows = [(chunk_id, repo_id, kind, vector_blob)];返回 [(lance_id, chunk_id)]。"""
        if not rows:
            return []
        data = [
            {
                "id": str(uuid.uuid4()),
                "vector": unpack_vector(blob),
                "chunk_id": chunk_id,
                "repo_id": repo_id,
                "kind": kind,
            }
            for chunk_id, repo_id, kind, blob in rows
        ]
        self.table.add(data)
        ids = [(d["id"], d["chunk_id"]) for d in data]
        return ids

    def delete_by_chunk_ids(self, chunk_ids: list[int]) -> None:
        if not chunk_ids:
            return
        ids = ",".join(str(int(i)) for i in chunk_ids)
        self.table.delete(f"chunk_id IN ({ids})")

    def count(self) -> int:
        try:
            return self.table.count_rows()
        except Exception:
            return 0

    def maybe_create_index(self) -> bool:
        """行数足够时建 ANN 索引;小语料跳过(暴力扫描即可)。"""
        if self.count() < MIN_ROWS_FOR_INDEX:
            return False
        try:
            self.table.create_index(metric="cosine")
            return True
        except Exception:
            return False
