import { NextResponse } from "next/server";

import {
  deleteChunk,
  getChunk,
  upsertChunk,
} from "@/app/api/knowledge/service";

export const runtime = "nodejs";

type Params = {
  params: {
    collection: string;
    chunkId: string;
  };
};

export async function GET(_: Request, { params }: Params) {
  try {
    const chunk = await getChunk(
      decodeURIComponent(params.collection),
      decodeURIComponent(params.chunkId),
    );

    if (!chunk) {
      return NextResponse.json(
        { error: "未找到指定 Chunk" },
        { status: 404 },
      );
    }

    return NextResponse.json({ data: chunk }, { status: 200 });
  } catch (error) {
    return handleError(error);
  }
}

export async function PATCH(request: Request, { params }: Params) {
  try {
    const collectionName = decodeURIComponent(params.collection);
    const chunkId = decodeURIComponent(params.chunkId);
    const existing = await getChunk(collectionName, chunkId);

    if (!existing) {
      return NextResponse.json(
        { error: "未找到指定 Chunk" },
        { status: 404 },
      );
    }

    const body = await request.json();
    const textCandidate =
      typeof body.text === "string" ? body.text.trim() : existing.text;
    const title =
      typeof body.title === "string" ? body.title.trim() : existing.title;
    const source =
      typeof body.source === "string" ? body.source.trim() : existing.source;
    const tags =
      Array.isArray(body.tags) && body.tags.every((tag: unknown) => typeof tag === "string")
        ? (body.tags as string[])
        : existing.tags ?? [];
    const metadata =
      body.metadata &&
      typeof body.metadata === "object" &&
      !Array.isArray(body.metadata)
        ? body.metadata
        : existing.metadata ?? {};

    if (!textCandidate) {
      return NextResponse.json(
        { error: "Chunk 内容不能为空" },
        { status: 400 },
      );
    }

    const data = await upsertChunk(collectionName, {
      id: chunkId,
      text: textCandidate,
      title,
      source,
      tags,
      metadata,
    });

    return NextResponse.json({ data }, { status: 200 });
  } catch (error) {
    return handleError(error);
  }
}

export async function DELETE(_: Request, { params }: Params) {
  try {
    const collectionName = decodeURIComponent(params.collection);
    const chunkId = decodeURIComponent(params.chunkId);
    await deleteChunk(collectionName, chunkId);
    return NextResponse.json({ success: true }, { status: 200 });
  } catch (error) {
    return handleError(error);
  }
}

function handleError(error: unknown, status = 500) {
  console.error(error);
  const message =
    error instanceof Error ? error.message : "未知错误，请稍后再试";
  return NextResponse.json({ error: message }, { status });
}
