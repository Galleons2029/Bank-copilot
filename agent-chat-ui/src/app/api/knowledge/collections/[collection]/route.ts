import { NextResponse } from "next/server";

import {
  deleteKnowledgeBase,
  getKnowledgeBase,
  upsertMetadata,
} from "@/app/api/knowledge/service";

export const runtime = "nodejs";

type Params = {
  params: {
    collection: string;
  };
};

export async function GET(_: Request, { params }: Params) {
  try {
    const collectionName = decodeURIComponent(params.collection);
    const data = await getKnowledgeBase(collectionName);
    return NextResponse.json({ data }, { status: 200 });
  } catch (error) {
    return handleError(error);
  }
}

export async function PATCH(request: Request, { params }: Params) {
  try {
    const collectionName = decodeURIComponent(params.collection);
    const body = await request.json();
    const displayName =
      typeof body.displayName === "string"
        ? body.displayName.trim()
        : undefined;
    const description =
      typeof body.description === "string"
        ? body.description.trim()
        : undefined;
    const tags =
      Array.isArray(body.tags) && body.tags.every((tag: unknown) => typeof tag === "string")
        ? (body.tags as string[])
        : undefined;

    if (displayName !== undefined && displayName.length === 0) {
      return NextResponse.json(
        { error: "展示名称不能为空" },
        { status: 400 },
      );
    }

    await upsertMetadata(collectionName, {
      displayName,
      description,
      tags,
    });

    const data = await getKnowledgeBase(collectionName);
    return NextResponse.json({ data }, { status: 200 });
  } catch (error) {
    return handleError(error);
  }
}

export async function DELETE(_: Request, { params }: Params) {
  try {
    const collectionName = decodeURIComponent(params.collection);
    await deleteKnowledgeBase(collectionName);
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
