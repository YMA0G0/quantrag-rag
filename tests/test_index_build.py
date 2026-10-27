"""Offline tests for the index builder CLI and index loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantrag.config import load_rag_settings
from quantrag.errors import ConfigurationError
from quantrag.index_build import build_index, main
from quantrag.service import RagService, build_embedder, build_service, load_index_embedder

pytestmark = pytest.mark.anyio

FIXTURE = Path(__file__).parent / "fixtures" / "mini_chunks.jsonl"


def anyio_backend() -> str:
    return "asyncio"


class FakeChat:
    def __init__(self, reply: str = "Indexed answer [1].") -> None:
        self.reply = reply

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        return self.reply


async def test_build_index_writes_all_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "index"

    manifest = await build_index(
        chunks_path=FIXTURE,
        out_dir=out,
        embedder_kind="local",
        batch_size=7,
    )

    assert manifest["format"] == 1
    assert manifest["chunks_total"] == 30
    assert manifest["chunks_indexed"] == 30
    assert manifest["batch_size"] == 7
    assert manifest["embedder"]["kind"] == "local"
    assert manifest["embedder"]["dimension"] > 0
    assert manifest["created_at"].endswith("+00:00")

    for name in ("manifest.json", "chunks.jsonl", "vectors.json", "embedder.json"):
        assert (out / name).is_file(), name

    rows = [
        json.loads(line)
        for line in (out / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 30
    assert rows[0]["id"] == "3460fe14fd-0000"
    assert rows[0]["text"].startswith("18.642 Lecture 19_1")  # body copied verbatim
    assert rows[0]["kind"] == "prose"

    vectors = json.loads((out / "vectors.json").read_text(encoding="utf-8"))
    assert len(vectors["chunks"]) == len(vectors["vectors"]) == 30

    state = json.loads((out / "embedder.json").read_text(encoding="utf-8"))
    assert len(state["vocab"]) == len(state["idf"]) == manifest["embedder"]["dimension"]


async def test_loaded_index_answers_questions(tmp_path: Path) -> None:
    out = tmp_path / "index"
    await build_index(chunks_path=FIXTURE, out_dir=out, embedder_kind="local")

    settings = load_rag_settings({"QUANTRAG_INDEX_DIR": str(out)}, load_env_file=False)
    chat = FakeChat("Brownian motion has drift [1].")
    service = RagService.from_index(
        out,
        chat=chat,
        settings=settings,
        embedder=load_index_embedder(out),
    )

    assert service.size == 30
    assert service.has_vectors is True
    assert service.corpus_label == str(out)

    hits = await service.search("Brownian motion with drift", k=3, mode="vector")
    assert len(hits) == 3

    result = await service.ask("What is Brownian motion with drift?")
    assert result.grounded is True
    assert result.sources[0]["cited"] is True


async def test_index_embedder_state_matches_the_build(tmp_path: Path) -> None:
    out = tmp_path / "index"
    await build_index(chunks_path=FIXTURE, out_dir=out, embedder_kind="local")

    from quantrag.retriever import load_chunks_jsonl

    rebuilt = build_embedder("local", chunks=load_chunks_jsonl(FIXTURE))
    restored = load_index_embedder(out)

    [expected] = await rebuilt.embed(["Brownian motion with drift"])
    [actual] = await restored.embed(["Brownian motion with drift"])

    assert actual == pytest.approx(expected)


async def test_build_service_prefers_the_index(tmp_path: Path) -> None:
    out = tmp_path / "index"
    await build_index(chunks_path=FIXTURE, out_dir=out, embedder_kind="local")

    settings = load_rag_settings(
        {"QUANTRAG_INDEX_DIR": str(out), "QUANTRAG_CORPUS_PATH": str(FIXTURE)},
        load_env_file=False,
    )
    service = await build_service(chat=FakeChat(), settings=settings)

    assert service.corpus_label == str(out)
    assert service.has_vectors is True
    assert service.size == 30


async def test_build_service_falls_back_to_the_corpus(tmp_path: Path) -> None:
    settings = load_rag_settings(
        {
            "QUANTRAG_INDEX_DIR": str(tmp_path / "missing-index"),
            "QUANTRAG_CORPUS_PATH": str(FIXTURE),
            "QUANTRAG_EMBEDDER": "local",
        },
        load_env_file=False,
    )

    service = await build_service(chat=FakeChat(), settings=settings, env={})

    assert service.corpus_label == str(FIXTURE)
    assert service.has_vectors is True
    assert service.size == 30


async def test_build_service_requires_a_knowledge_base(tmp_path: Path) -> None:
    settings = load_rag_settings(
        {
            "QUANTRAG_INDEX_DIR": str(tmp_path / "missing-index"),
            "QUANTRAG_CORPUS_PATH": str(tmp_path / "missing-chunks.jsonl"),
        },
        load_env_file=False,
    )

    with pytest.raises(ConfigurationError):
        await build_service(chat=FakeChat(), settings=settings, env={})


async def test_rebuilding_requires_force(tmp_path: Path) -> None:
    out = tmp_path / "index"
    await build_index(chunks_path=FIXTURE, out_dir=out, embedder_kind="local")

    with pytest.raises(ConfigurationError):
        await build_index(chunks_path=FIXTURE, out_dir=out, embedder_kind="local")

    manifest = await build_index(
        chunks_path=FIXTURE,
        out_dir=out,
        embedder_kind="local",
        force=True,
    )
    assert manifest["chunks_indexed"] == 30


async def test_build_index_rejects_invalid_input(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        await build_index(
            chunks_path=tmp_path / "nope.jsonl",
            out_dir=tmp_path / "index",
            embedder_kind="local",
        )

    with pytest.raises(ConfigurationError):
        await build_index(
            chunks_path=FIXTURE,
            out_dir=tmp_path / "index",
            embedder_kind="local",
            batch_size=0,
        )


def test_cli_builds_an_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    out = tmp_path / "index"
    monkeypatch.setenv("QUANTRAG_CORPUS_PATH", str(FIXTURE))
    monkeypatch.setenv("QUANTRAG_INDEX_DIR", str(out))
    monkeypatch.setenv("QUANTRAG_EMBEDDER", "local")
    capsys.readouterr()

    assert main([]) == 0

    captured = capsys.readouterr()
    assert "indexed 30/30 chunks" in captured.out
    assert (out / "manifest.json").is_file()


def test_cli_reports_missing_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("QUANTRAG_CORPUS_PATH", str(tmp_path / "missing.jsonl"))
    monkeypatch.setenv("QUANTRAG_INDEX_DIR", str(tmp_path / "index"))
    capsys.readouterr()

    assert main([]) == 1

    captured = capsys.readouterr()
    assert captured.err.startswith("error:")
