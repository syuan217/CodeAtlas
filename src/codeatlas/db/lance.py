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
        """rows = [(chunk_id, repo_id, kind, vector_blob)];返回 [(lance_id, chunk_id)]。

        幂等防御:写入前先删同 chunk_id 的旧行。SQLite 会复用被级联删除的
        chunk id(无 AUTOINCREMENT),中断重跑/自愈重处理若残留旧向量行,
        同 id 会叠加成重复行污染检索,故每次写入先清理。
        """
        if not rows:
            return []
        self.delete_by_chunk_ids([r[0] for r in rows])
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
        """行数足够且尚无 ANN 索引时才建;已存在则跳过。

        IVF_PQ 全量训练在无 GPU 的 Intel 机器上耗时可达分钟级,绝不能每轮重建
        (曾导致无变更的二次运行耗时 8 分钟)。数据大幅增长后的重建留给
        显式维护入口,这里不做。
        """
        if self.count() < MIN_ROWS_FOR_INDEX:
            return False
        try:
            if self.table.list_indices():  # 已有任何索引 → 不重复建
                return False
            self.table.create_index(metric="cosine")
            return True
        except Exception:
            return False

    def reconcile_orphans(self, valid_chunk_ids: set[int], expected_rows: int | None = None) -> int:
        """删除 Lance 中不属于任何现存 chunk 的孤儿向量(中断残留)。

        expected_rows(vector_refs 行数)一致时直接跳过,避免每轮全表扫描;
        不一致才 to_arrow 拉全表对账。TODO 百万 chunk 级需换成分片对账。
        """
        try:
            if expected_rows is not None and self.count() == expected_rows:
                return 0
            arrow = self.table.to_arrow()  # pylance 不可用,to_arrow 全表(大库慎用)
            all_ids = set(arrow.column("chunk_id").to_pylist())
        except Exception:
            return 0
        orphans = sorted(i for i in all_ids if i not in valid_chunk_ids)
        if not orphans:
            return 0
        BATCH = 5000
        for i in range(0, len(orphans), BATCH):
            ids = ",".join(str(int(x)) for x in orphans[i : i + BATCH])
            self.table.delete(f"chunk_id IN ({ids})")
        return len(orphans)

    def rebuild_from_sql(self, conn) -> tuple[int, int]:
        """一次性修复:按 vector_refs + embed_cache 重写整张向量表。

        返回 (重建行数, 缺向量行数)。缺向量 = chunks 没有对应 embed_cache
        条目(换模型/缓存被清),需要重新 embedding,调用方负责提示。
        vector_refs 的 lance_id 同步更新。
        """
        import sqlite3

        assert isinstance(conn, sqlite3.Connection)
        rows = conn.execute(
            "SELECT v.chunk_id AS chunk_id, c.repo_id AS repo_id, c.kind AS kind, "
            "       c.content_hash AS content_hash, v.model AS model "
            "FROM vector_refs v JOIN chunks c ON v.chunk_id = c.id"
        ).fetchall()
        data = []
        missing = 0
        for r in rows:
            cached = conn.execute(
                "SELECT vector FROM embed_cache WHERE content_hash=? AND model=?",
                (r["content_hash"], r["model"]),
            ).fetchone()
            if cached is None:
                missing += 1
                continue
            data.append(
                (r["chunk_id"], r["repo_id"], r["kind"], cached["vector"])
            )
        # 覆盖重建:删表重建,杜绝一切历史重复/孤儿
        self.db.drop_table("chunks")
        self._table = None
        if data:
            new_ids = self.add_vectors(data)
            conn.executemany(
                "UPDATE vector_refs SET lance_id=? WHERE chunk_id=?",
                [(lance_id, chunk_id) for lance_id, chunk_id in new_ids],
            )
        return len(data), missing
