"""Offline ingestion pipeline for the QuantRAG knowledge base.

This package is deliberately separate from the server runtime: ``server.py``
must never import from it.  It turns source CSVs into ``chunks.jsonl``.
"""
