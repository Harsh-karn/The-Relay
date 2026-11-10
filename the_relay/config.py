"""Configuration management for The Relay."""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load local .env if present
load_dotenv(override=False)


class RelayConfig(BaseModel):
    """Runtime configuration for Groq and Stand-in models."""

    groq_api_key: str = Field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )
    groq_model: str = Field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    standin_host: str = Field(
        default_factory=lambda: os.getenv("STANDIN_HOST", "127.0.0.1")
    )
    standin_port: int = Field(
        default_factory=lambda: int(os.getenv("STANDIN_PORT", "8000"))
    )

    # Project directories
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    results_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "results")

    @property
    def has_groq_key(self) -> bool:
        return bool(self.groq_api_key and not self.groq_api_key.startswith("your_groq_api_key"))


config = RelayConfig()

