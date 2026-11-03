from google import genai
from google.genai import types
from app.core.config import settings


class Embedding:

    def __init__(self, model_name: str = "gemini-embedding-001"):
        self.model_name = model_name
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    async def embed_batch(self, chunks: list[str]) -> list[list[float]]:

        if not chunks:
            return []

        result = await self.client.aio.models.embed_content(
            model=self.model_name,
            contents=chunks,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )

        return [item.values for item in result.embeddings]

    async def embed_chunks(
        self,
        chunks: list[str],
        batch_size: int = 32
    ) -> list[list[float]]:

        if not chunks:
            return []

        all_embeddings = []

        for i in range(0, len(chunks), batch_size):

            batch = chunks[i:i + batch_size]

            embeddings = await self.embed_batch(batch)

            all_embeddings.extend(embeddings)

        return all_embeddings

    async def embed_query(self, prompt: str) -> list[float]:

        if not prompt.strip():
            return []

        result = await self.client.aio.models.embed_content(
            model=self.model_name,
            contents=prompt,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY"
            )
        )

        return result.embeddings[0].values


embedding = Embedding()

# import asyncio
# from sentence_transformers import SentenceTransformer


# class Embedding:
#     def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
#         self.model_name = model_name
#         try:
#             self.model = SentenceTransformer(model_name, local_files_only=True)
#         except Exception:
#             self.model = SentenceTransformer(model_name)

#     async def embed_chunks(self, chunks: list[str], batch_size: int = 32) -> list[list[float]]:
#         if not chunks:
#             return []
            
#         embeddings = await asyncio.to_thread(self.model.encode, chunks, batch_size=batch_size)
#         return embeddings.tolist()

#     async def embed_query(self, prompt: str) -> list[float]:
#         embedding = await asyncio.to_thread(self.model.encode, prompt)
#         return embedding.tolist()


# embedding = Embedding()
