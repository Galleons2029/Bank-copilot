"""Knowledge base management API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.api.services import knowledge as knowledge_service
from app.models.knowledge import (
    ChunkCreateRequest,
    ChunkListResponse,
    ChunkResponse,
    ChunkUpdateRequest,
    KnowledgeBaseCreateRequest,
    KnowledgeBaseListResponse,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdateRequest,
    SuccessResponse,
)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("/collections", response_model=KnowledgeBaseListResponse)
async def list_collections() -> KnowledgeBaseListResponse:
    data = await knowledge_service.list_knowledge_bases()
    return KnowledgeBaseListResponse(data=data)


@router.post(
    "/collections",
    response_model=KnowledgeBaseResponse,
    status_code=201,
)
async def create_collection(
    payload: KnowledgeBaseCreateRequest,
) -> KnowledgeBaseResponse:
    data = await knowledge_service.create_knowledge_base(payload)
    return KnowledgeBaseResponse(data=data)


@router.get(
    "/collections/{collection}",
    response_model=KnowledgeBaseResponse,
)
async def get_collection(collection: str) -> KnowledgeBaseResponse:
    data = await knowledge_service.get_knowledge_base(collection)
    return KnowledgeBaseResponse(data=data)


@router.patch(
    "/collections/{collection}",
    response_model=KnowledgeBaseResponse,
)
async def update_collection(
    collection: str,
    payload: KnowledgeBaseUpdateRequest,
) -> KnowledgeBaseResponse:
    if payload.display_name is None and payload.description is None and payload.tags is None:
        raise HTTPException(status_code=400, detail="请至少提供一个需要更新的字段")

    await knowledge_service.upsert_metadata(collection, payload)
    data = await knowledge_service.get_knowledge_base(collection)
    return KnowledgeBaseResponse(data=data)


@router.delete(
    "/collections/{collection}",
    response_model=SuccessResponse,
)
async def delete_collection(collection: str) -> SuccessResponse:
    await knowledge_service.delete_knowledge_base(collection)
    return SuccessResponse(success=True)


@router.get(
    "/collections/{collection}/chunks",
    response_model=ChunkListResponse,
)
async def list_chunks(
    collection: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: str | int | None = Query(default=None),
) -> ChunkListResponse:
    chunks, next_offset = await knowledge_service.fetch_chunks(
        collection,
        limit=limit,
        offset=offset,
    )
    return ChunkListResponse(data=chunks, next_offset=next_offset)


@router.post(
    "/collections/{collection}/chunks",
    response_model=ChunkResponse,
    status_code=201,
)
async def create_chunk(
    collection: str,
    payload: ChunkCreateRequest,
) -> ChunkResponse:
    chunk = await knowledge_service.upsert_chunk(collection, payload)
    return ChunkResponse(data=chunk)


@router.get(
    "/collections/{collection}/chunks/{chunk_id}",
    response_model=ChunkResponse,
)
async def get_chunk(collection: str, chunk_id: str) -> ChunkResponse:
    chunk = await knowledge_service.get_chunk(collection, chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="未找到指定 Chunk")
    return ChunkResponse(data=chunk)


@router.patch(
    "/collections/{collection}/chunks/{chunk_id}",
    response_model=ChunkResponse,
)
async def update_chunk(
    collection: str,
    chunk_id: str,
    payload: ChunkUpdateRequest,
) -> ChunkResponse:
    existing = await knowledge_service.get_chunk(collection, chunk_id)
    if not existing:
        raise HTTPException(status_code=404, detail="未找到指定 Chunk")

    merged_payload = ChunkCreateRequest(
        text=payload.text or existing.text,
        title=payload.title if payload.title is not None else existing.title,
        source=payload.source if payload.source is not None else existing.source,
        tags=payload.tags if payload.tags is not None else (existing.tags or []),
        metadata=payload.metadata if payload.metadata is not None else existing.metadata or {},
    )

    chunk = await knowledge_service.upsert_chunk(collection, merged_payload, chunk_id=chunk_id)
    return ChunkResponse(data=chunk)


@router.delete(
    "/collections/{collection}/chunks/{chunk_id}",
    response_model=SuccessResponse,
)
async def delete_chunk(collection: str, chunk_id: str) -> SuccessResponse:
    await knowledge_service.delete_chunk(collection, chunk_id)
    return SuccessResponse(success=True)
