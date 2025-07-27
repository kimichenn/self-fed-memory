"""CLI helper - bulk-ingest an entire directory of ``.md`` files.

Usage::

    python -m scripts.ingest_folder ~/Notes
"""

from __future__ import annotations

import pathlib

import click

from app.core.embeddings import get_embeddings
from app.core.memory import MemoryManager
from app.ingestion.markdown_loader import parse_markdown_file


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path)
)
@click.option("--dry-run", is_flag=True, help="Parse + chunk but skip Pinecone upload.")
def main(
    directory: pathlib.Path, dry_run: bool, memory_manager: MemoryManager | None = None
) -> None:
    """Index every ``.md`` under *DIRECTORY* (recursive)."""
    if memory_manager:
        manager = memory_manager
    elif dry_run:
        manager = None
    else:
        embeddings = get_embeddings()
        manager = MemoryManager(embeddings)

    paths = list(directory.rglob("*.md"))
    if not paths:
        click.echo("No markdown files found - exiting.")
        raise SystemExit(0)

    with click.progressbar(paths, label="Ingesting files...") as bar:
        for p in bar:
            chunks = parse_markdown_file(p)
            if manager:
                manager.add_chunks(chunks)

    click.echo("Done.")


if __name__ == "__main__":  # pragma: no cover
    main()
