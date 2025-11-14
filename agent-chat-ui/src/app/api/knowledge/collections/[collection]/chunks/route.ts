import { NextResponse } from "next/server";

import {
  fetchChunks,
  upsertChunk,
} from "@/app/api/knowledge/service";

export const runtime = "nodejs";

type Params = {
  params: {
    collection: string;
  };
};

export async function GET(request: Request, { params }: Params) {
  try {
    const collectionName = decodeURIComponent(params.collection);
    const { searchParams } = new URL(request.url);
    const limit = Math.max(
      1,
      Math.min(Number(searchParams.get("limit") ?? 50), 200),
    );
    const offsetParam = searchParams.get("offset");
    const offset = parseOffset(offsetParam);

    const { chunks, nextOffset } = await fetchChunks(collectionName, {
      limit,
      offset,
    });

    return NextResponse.json(
      { data: chunks, nextOffset },
      { status: 200 },
    );
  } catch (error) {
    return handleError(error);
  }
}

export async function POST(request: Request, { params }: Params) {
  try {
    const collectionName = decodeURIComponent(params.collection);
    const body = await request.json();
    const text = typeof body.text === "string" ? body.text.trim() : "";
    const title =
      typeof body.title === "string" ? body.title.trim() : undefined;
    const source =
      typeof body.source === "string" ? body.source.trim() : undefined;
    const tags =
      Array.isArray(body.tags) && body.tags.every((tag: unknown) => typeof tag === "string")
        ? body.tags
        : [];
    const metadata =
      body.metadata &&
      typeof body.metadata === "object" &&
      !Array.isArray(body.metadata)
        ? body.metadata
        : {};

    if (!text) {
      return NextResponse.json(
        { error: "Chunk 内容不能为空" },
        { status: 400 },
      );
    }

    const data = await upsertChunk(collectionName, {
      id: typeof body.id === "string" ? body.id : undefined,
      text,
      title,
      source,
      tags,
      metadata,
    });

    return NextResponse.json({ data }, { status: 201 });
  } catch (error) {
    return handleError(error);
  }
}

function parseOffset(value: string | null) {
  if (!value) {
    return undefined;
  }

  const numeric = Number(value);
  if (!Number.isNaN(numeric) && Number.isFinite(numeric)) {
    return numeric;
  }

  return value;
}

function handleError(error: unknown, status = 500) {
  console.error(error);
  const message =
    error instanceof Error ? error.message : "未知错误，请稍后再试";
  return NextResponse.json({ error: message }, { status });
}
