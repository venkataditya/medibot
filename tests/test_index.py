from qdrant_client import QdrantClient, models

from medibot import index
from medibot.ingestion import ChunkRecord


class FakeEmbedder:
    dense_dim = 4

    def embed_documents(self, texts: list[str]) -> list[index.Vectors]:
        return [self._vectors(t) for t in texts]

    def embed_query(self, text: str) -> index.Vectors:
        return self._vectors(text)

    @staticmethod
    def _vectors(text: str) -> index.Vectors:
        n = float(len(text))
        return index.Vectors(dense=[n, 1.0, 0.0, 0.0], sparse_indices=[len(text) % 7], sparse_values=[1.0])


def record(i: int, collection: str = "clinical") -> ChunkRecord:
    return ChunkRecord(
        text=f"Heading\nbody {i}",
        body=f"body {i}",
        source_document="drug_formulary.pdf",
        collection=collection,
        access_roles=["doctor", "admin"],
        section_title="Heading",
        chunk_type="text",
        heading_path=["Heading"],
        chunk_index=i,
    )


def test_ensure_collection_creates_dense_and_idf_weighted_sparse_vectors() -> None:
    client = QdrantClient(":memory:")
    index.ensure_collection(client, "docs", dense_dim=4)
    info = client.get_collection("docs")
    assert info.config.params.vectors[index.DENSE].size == 4
    assert info.config.params.vectors[index.DENSE].distance == models.Distance.COSINE
    assert info.config.params.sparse_vectors[index.SPARSE].modifier == models.Modifier.IDF


def test_ensure_collection_starts_from_empty_on_reingest() -> None:
    client = QdrantClient(":memory:")
    index.ensure_collection(client, "docs", dense_dim=4)
    index.index_records(client, "docs", [record(0)], FakeEmbedder())
    index.ensure_collection(client, "docs", dense_dim=4)
    assert client.count("docs").count == 0


def test_payload_carries_the_full_metadata_schema() -> None:
    payload = index.payload_for(record(3))
    assert payload == {
        "text": "Heading\nbody 3",
        "body": "body 3",
        "source_document": "drug_formulary.pdf",
        "collection": "clinical",
        "access_roles": ["doctor", "admin"],
        "section_title": "Heading",
        "chunk_type": "text",
        "heading_path": ["Heading"],
        "chunk_index": 3,
    }


def test_point_ids_are_stable_across_reingests() -> None:
    assert index.point_id(record(1)) == index.point_id(record(1))
    assert index.point_id(record(1)) != index.point_id(record(2))


def test_index_records_upserts_both_vectors_and_payload() -> None:
    client = QdrantClient(":memory:")
    index.ensure_collection(client, "docs", dense_dim=4)
    count = index.index_records(client, "docs", [record(0), record(1), record(2)], FakeEmbedder(), batch_size=2)
    assert count == 3
    assert client.count("docs").count == 3
    point = client.retrieve("docs", ids=[index.point_id(record(1))], with_vectors=True, with_payload=True)[0]
    assert point.payload["chunk_index"] == 1
    dense = point.vector[index.DENSE]
    assert len(dense) == 4 and dense[0] > 0 and dense[2] == 0  # cosine-normalised on insert
    assert point.vector[index.SPARSE].indices == [len("Heading\nbody 1") % 7]


def test_client_kwargs_prefer_url_over_local_path(tmp_path) -> None:
    assert index.client_kwargs(url="http://q:6333", path=tmp_path) == {"url": "http://q:6333"}
    assert index.client_kwargs(url=None, path=tmp_path) == {"path": str(tmp_path)}
