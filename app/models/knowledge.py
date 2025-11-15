"""
Pydantic models for knowledge base management APIs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _to_camel(string: str) -> str:
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class CamelModel(BaseModel):
    """Base model that serializes fields using camelCase."""

    model_config = ConfigDict(populate_by_name=True, alias_generator=_to_camel)


class KnowledgeMetadata(CamelModel):
    display_name: str = Field(..., min_length=1)
    description: str | None = None
    tags: list[str] = Field(default_factory=list)


class KnowledgeBase(CamelModel):
    name: str
    status: str
    vector_size: int = Field(..., alias="vectorSize")
    distance: str | None = None
    chunk_count: int = Field(..., alias="chunkCount", ge=0)
    metadata: KnowledgeMetadata


class KnowledgeBaseListResponse(CamelModel):
    data: list[KnowledgeBase]


class KnowledgeBaseResponse(CamelModel):
    data: KnowledgeBase


class KnowledgeBaseCreateRequest(CamelModel):
    name: str = Field(..., min_length=1, max_length=128)
    display_name: str | None = Field(default=None, min_length=1)
    description: str | None = None
    vector_size: int = Field(default=1536, alias="vectorSize", gt=0)
    distance: str = Field(default="Cosine")


class KnowledgeBaseUpdateRequest(CamelModel):
    display_name: str | None = Field(default=None, min_length=1)
    description: str | None = None
    tags: list[str] | None = None


class KnowledgeChunk(CamelModel):
    id: str
    text: str
    title: str | None = None
    source: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None
    updated_at: str | None = Field(default=None, alias="updatedAt")


class ChunkResponse(CamelModel):
    data: KnowledgeChunk


class ChunkListResponse(CamelModel):
    data: list[KnowledgeChunk]
    next_offset: str | int | None = Field(default=None, alias="nextOffset")


class ChunkCreateRequest(CamelModel):
    text: str = Field(..., min_length=1)
    title: str | None = None
    source: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkUpdateRequest(CamelModel):
    text: str | None = Field(default=None)
    title: str | None = None
    source: str | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None


class SuccessResponse(CamelModel):
    success: bool = True
