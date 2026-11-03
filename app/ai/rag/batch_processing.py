
import asyncio
import logging
from pathlib import Path
from typing import Callable, Coroutine, Any

from app.core.config import settings
from app.ai.rag.embed import embedding

logger = logging.getLogger("uvicorn.error")


class BatchProcessor:

    def __init__(
        self,
        embedding_batch_size: int = settings.EMBEDDING_BATCH_SIZE,
        ingestion_batch_size: int = settings.INGESTION_BATCH_SIZE,
        max_concurrency: int = settings.MAX_CONCURRENCY,
        rate_limit_delay: float = settings.RATE_LIMIT_DELAY,
    ):
        self.embedding_batch_size = embedding_batch_size
        self.ingestion_batch_size = ingestion_batch_size
        self.max_concurrency = max_concurrency
        self.rate_limit_delay = rate_limit_delay

    @staticmethod
    def chunk_list(items: list, size: int) -> list[list]:
        size = max(1, size)
        return [items[i:i + size] for i in range(0, len(items), size)]

    async def embed_chunks_batched(
        self,
        chunks: list[dict] | list[str],
        batch_size: int | None = None,
        rate_limit_delay: float | None = None,
        **kwargs,
    ) -> list[list[float]]:

        if not chunks:
            return []

        chunk_texts = [
            c["text"] if isinstance(c, dict) else c
            for c in chunks
        ]

        batch_size = batch_size or self.embedding_batch_size
        delay = rate_limit_delay if rate_limit_delay is not None else self.rate_limit_delay

        batches = self.chunk_list(chunk_texts, batch_size)
        total_batches = len(batches)

        all_embeddings = []

        logger.info("=" * 50)
        logger.info("EMBEDDING BATCH PROCESSING STARTED")
        logger.info(f"Total chunks      : {len(chunk_texts)}")
        logger.info(f"Batch size        : {batch_size}")
        logger.info(f"Total batches     : {total_batches}")
        logger.info("=" * 50)

        for index, batch in enumerate(batches, start=1):

            logger.info(
                f"[Embedding Batch {index}/{total_batches}] "
                f"Processing {len(batch)} chunks..."
            )

            embeddings = await embedding.embed_chunks(
                batch,
                batch_size=len(batch)
            )

            all_embeddings.extend(embeddings)

            logger.info(
                f"[Embedding Batch {index}/{total_batches}] "
                f"Completed: {len(embeddings)} embeddings generated"
            )

            if delay > 0 and index < total_batches:
                await asyncio.sleep(delay)

        logger.info("EMBEDDING BATCH PROCESSING COMPLETED")
        logger.info(f"Total embeddings generated: {len(all_embeddings)}")

        return all_embeddings

    async def process_ingestion_batches(
        self,
        files: list[Path],
        ingest_func: Callable[[Path], Coroutine[Any, Any, dict]],
        batch_size: int | None = None,
        max_concurrency: int | None = None,
        rate_limit_delay: float | None = None,
    ) -> list[dict]:

        if not files:
            return []

        batch_size = batch_size or self.ingestion_batch_size
        concurrency = max_concurrency or self.max_concurrency
        delay = rate_limit_delay if rate_limit_delay is not None else self.rate_limit_delay

        semaphore = asyncio.Semaphore(concurrency)
        batches = self.chunk_list(files, batch_size)

        total_batches = len(batches)
        all_results = []

        logger.info("=" * 50)
        logger.info("DOCUMENT INGESTION STARTED")
        logger.info(f"Total files       : {len(files)}")
        logger.info(f"Batch size        : {batch_size}")
        logger.info(f"Total batches     : {total_batches}")
        logger.info(f"Max concurrency   : {concurrency}")
        logger.info("=" * 50)

        async def ingest_single(file_path: Path):
            async with semaphore:
                try:
                    result = await ingest_func(file_path)
                    return {"file": file_path.name, **result}

                except Exception as e:
                    logger.exception(f"Failed to ingest {file_path.name}")
                    return {
                        "file": file_path.name,
                        "status": "failed",
                        "message": str(e),
                        "document": None
                    }

        for index, batch in enumerate(batches, start=1):

            logger.info(
                f"[Ingestion Batch {index}/{total_batches}] "
                f"Processing {len(batch)} files..."
            )

            results = await asyncio.gather(
                *(ingest_single(file) for file in batch)
            )

            all_results.extend(results)

            successful = sum(
                1 for result in results
                if result.get("status") != "failed"
            )

            logger.info(
                f"[Ingestion Batch {index}/{total_batches}] "
                f"Completed: {successful}/{len(batch)} successful"
            )

            if delay > 0 and index < total_batches:
                await asyncio.sleep(delay)

        logger.info("DOCUMENT INGESTION COMPLETED")
        logger.info(f"Total files processed: {len(all_results)}")

        return all_results


batch_processor = BatchProcessor()


# import asyncio
# import logging
# from pathlib import Path
# from typing import Callable, Coroutine, Any
# from app.core.config import settings
# from app.ai.rag.embed import embedding

# logger = logging.getLogger("uvicorn.error")


# class BatchProcessor:
#     """
#     Handles batched operations for the RAG pipeline, including:
#     - Batched embedding generation with concurrency control and rate limiting.
#     - Batched document ingestion with error handling and concurrency.
#     - Preserves chunk metadata and 1:1 chunk-to-embedding mapping.
#     """

#     def __init__(
#         self,
#         embedding_batch_size: int = settings.EMBEDDING_BATCH_SIZE,
#         ingestion_batch_size: int = settings.INGESTION_BATCH_SIZE,
#         max_concurrency: int = settings.MAX_CONCURRENCY,
#         rate_limit_delay: float = settings.RATE_LIMIT_DELAY,
#     ):
#         self.embedding_batch_size = embedding_batch_size
#         self.ingestion_batch_size = ingestion_batch_size
#         self.max_concurrency = max_concurrency
#         self.rate_limit_delay = rate_limit_delay

#     @staticmethod
#     def chunk_list(items: list, size: int) -> list[list]:
#         """Splits a list into smaller batches of the specified size."""
#         if size <= 0:
#             size = 1
#         return [items[i:i + size] for i in range(0, len(items), size)]

#     async def embed_chunks_batched(
#         self,
#         chunks: list[dict] | list[str],
#         batch_size: int | None = None,
#         rate_limit_delay: float | None = None,
#         **kwargs,
#     ) -> list[list[float]]:
#         """
#         Generates embeddings for chunks in batches while preserving chunk order and mapping.

#         - Supports configurable batch sizes.
#         - Processes batches sequentially: SentenceTransformer.encode is CPU-bound and
#           runs in a thread via asyncio.to_thread; concurrent thread submissions for a
#           local model cause thread pool exhaustion, so sequential processing is both
#           correct and optimal here.
#         - Applies rate limiting between batch requests to avoid overwhelming the provider.
#         - Leverages provider batch APIs (SentenceTransformer encode batch_size).
#         - Preserves strict 1:1 mapping between input chunks and returned embeddings.
#         """
#         if not chunks:
#             return []

#         # Extract text while preserving mapping order
#         chunk_texts = [
#             c["text"] if isinstance(c, dict) else c
#             for c in chunks
#         ]

#         effective_batch_size = batch_size or self.embedding_batch_size
#         effective_delay = rate_limit_delay if rate_limit_delay is not None else self.rate_limit_delay

#         batches = self.chunk_list(chunk_texts, effective_batch_size)
#         total_batches = len(batches)
#         logger.info(
#             f"Generating embeddings for {len(chunk_texts)} chunks across {total_batches} "
#             f"batches (batch_size={effective_batch_size})"
#         )

#         # Process batches sequentially to preserve order and avoid thread pool saturation
#         # for CPU-bound local embedding models
#         all_embeddings: list[list[float]] = []
#         for batch_idx, batch_texts in enumerate(batches):
#             # Apply rate limiting between batches (skip delay on the first batch)
#             if effective_delay > 0 and batch_idx > 0:
#                 await asyncio.sleep(effective_delay)

#             # Provider batch API call — passes the batch to encode in one call
#             batch_embeddings = await embedding.embed_chunks(batch_texts, batch_size=len(batch_texts))
#             all_embeddings.extend(batch_embeddings)

#         return all_embeddings

#     async def process_ingestion_batches(
#         self,
#         files: list[Path],
#         ingest_func: Callable[[Path], Coroutine[Any, Any, dict]],
#         batch_size: int | None = None,
#         max_concurrency: int | None = None,
#         rate_limit_delay: float | None = None,
#     ) -> list[dict]:
#         """
#         Processes document files in batches with concurrency and rate limiting.

#         Files within each batch run concurrently (bounded by max_concurrency semaphore)
#         since file I/O and hash checks are I/O-bound; batches themselves run sequentially
#         so rate limiting applies between them.
#         """
#         if not files:
#             return []

#         effective_batch_size = batch_size or self.ingestion_batch_size
#         effective_delay = rate_limit_delay if rate_limit_delay is not None else self.rate_limit_delay
#         semaphore = asyncio.Semaphore(max_concurrency or self.max_concurrency)

#         file_batches = self.chunk_list(files, effective_batch_size)
#         all_results: list[dict] = []

#         for batch_idx, batch_files in enumerate(file_batches):
#             # Apply rate limiting between file batches
#             if effective_delay > 0 and batch_idx > 0:
#                 await asyncio.sleep(effective_delay)

#             # Use default-argument capture to avoid late-binding closure issues
#             async def _ingest_single(file_path: Path = None):
#                 async with semaphore:
#                     try:
#                         result = await ingest_func(file_path)
#                     except Exception as e:
#                         logger.exception(f"Failed to ingest '{file_path.name}': {e}")
#                         result = {"status": "failed", "message": str(e), "document": None}
#                     return {"file": file_path.name, **result}

#             batch_tasks = [_ingest_single(p) for p in batch_files]
#             batch_results = await asyncio.gather(*batch_tasks)
#             all_results.extend(batch_results)

#         return all_results


# batch_processor = BatchProcessor()
