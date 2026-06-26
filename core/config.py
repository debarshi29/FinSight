from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "finsight_chunks"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    chunk_size: int = 400
    chunk_overlap: int = 80
    retrieval_top_k: int = 20
    rerank_top_k: int = 5

    confidence_threshold: float = 0.65
    hallucination_fallback_threshold: float = 0.50

    audit_log_dir: str = "audit_logs"

    otel_enabled: bool = False
    otel_endpoint: str = "http://localhost:4317"

    log_level: str = "INFO"


settings = Settings()
