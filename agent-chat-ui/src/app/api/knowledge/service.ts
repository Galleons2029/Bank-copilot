import { resolveApiUrl } from "@/lib/langgraph-config";
import { KnowledgeBase, KnowledgeChunk, KnowledgeMetadata } from "@/types/knowledge";

const API_PREFIX = "/api/v1/knowledge";
const BACKEND_URL =
  process.env.BACKEND_API_URL?.replace(/\/$/, "") ||
  resolveApiUrl().replace(/\/$/, "");

type KnowledgeBaseListPayload = {
  data?: KnowledgeBase[];
};

type KnowledgeBasePayload = {
  data: KnowledgeBase;
};

type ChunkListPayload = {
  data?: KnowledgeChunk[];
  nextOffset?: string | number;
};

type ChunkPayload = {
  data: KnowledgeChunk;
};

type SuccessPayload = {
  success: boolean;
};

function buildUrl(path: string): string {
  return `${BACKEND_URL}${path}`;
}

async function requestBackend<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(buildUrl(path), {
    ...init,
    headers: {
      "content-type": "application/json",
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });

  const payload = await parseResponse(response);
  if (!response.ok) {
    const message = extractErrorMessage(payload) ?? response.statusText;
    throw new Error(message || "请求失败");
  }

  return (payload as T) ?? ({} as T);
}

async function parseResponse(response: Response) {
  if (response.status === 204) {
    return null;
  }
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return response.text();
}

function extractErrorMessage(payload: unknown): string | undefined {
  if (!payload) {
    return undefined;
  }
  if (typeof payload === "string") {
    return payload;
  }
  if (typeof payload === "object") {
    if ("detail" in payload && typeof payload.detail === "string") {
      return payload.detail;
    }
    if ("error" in payload && typeof payload.error === "string") {
      return payload.error;
    }
  }
  return undefined;
}

export async function listKnowledgeBases(): Promise<KnowledgeBase[]> {
  const result = await requestBackend<KnowledgeBaseListPayload>(`${API_PREFIX}/collections`);
  return Array.isArray(result.data) ? result.data : [];
}

export async function getKnowledgeBase(name: string): Promise<KnowledgeBase> {
  const result = await requestBackend<KnowledgeBasePayload>(
    `${API_PREFIX}/collections/${encodeURIComponent(name)}`,
  );
  return result.data;
}

export async function createKnowledgeBase(input: {
  name: string;
  vectorSize: number;
  distance?: string;
  description?: string;
  displayName?: string;
}): Promise<KnowledgeBase> {
  const result = await requestBackend<KnowledgeBasePayload>(`${API_PREFIX}/collections`, {
    method: "POST",
    body: JSON.stringify(input),
  });
  return result.data;
}

export async function deleteKnowledgeBase(name: string) {
  await requestBackend<SuccessPayload>(`${API_PREFIX}/collections/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}

export async function upsertMetadata(collection: string, metadata: Partial<KnowledgeMetadata>) {
  await requestBackend<KnowledgeBasePayload>(`${API_PREFIX}/collections/${encodeURIComponent(collection)}`, {
    method: "PATCH",
    body: JSON.stringify(metadata),
  });
}

export async function fetchChunks(
  collection: string,
  options: { limit?: number; offset?: string | number } = {},
) {
  const params = new URLSearchParams();
  if (options.limit) {
    params.set("limit", String(options.limit));
  }
  if (options.offset !== undefined && options.offset !== null) {
    params.set("offset", String(options.offset));
  }

  const query = params.toString();
  const result = await requestBackend<ChunkListPayload>(
    `${API_PREFIX}/collections/${encodeURIComponent(collection)}/chunks${query ? `?${query}` : ""}`,
  );
  return {
    chunks: Array.isArray(result.data) ? result.data : [],
    nextOffset: result.nextOffset,
  };
}

export async function upsertChunk(
  collection: string,
  input: {
    id?: string;
    text: string;
    title?: string;
    source?: string;
    tags?: string[];
    metadata?: Record<string, unknown>;
  },
): Promise<KnowledgeChunk> {
  const payload = {
    text: input.text,
    title: input.title,
    source: input.source,
    tags: input.tags ?? [],
    metadata: input.metadata ?? {},
  };
  const encodedCollection = encodeURIComponent(collection);

  const result = input.id
    ? await requestBackend<ChunkPayload>(
        `${API_PREFIX}/collections/${encodedCollection}/chunks/${encodeURIComponent(input.id)}`,
        {
          method: "PATCH",
          body: JSON.stringify(payload),
        },
      )
    : await requestBackend<ChunkPayload>(`${API_PREFIX}/collections/${encodedCollection}/chunks`, {
        method: "POST",
        body: JSON.stringify(payload),
      });

  return result.data;
}

export async function deleteChunk(collection: string, chunkId: string) {
  await requestBackend<SuccessPayload>(
    `${API_PREFIX}/collections/${encodeURIComponent(collection)}/chunks/${encodeURIComponent(chunkId)}`,
    {
      method: "DELETE",
    },
  );
}

export async function getChunk(collection: string, chunkId: string) {
  const result = await requestBackend<ChunkPayload>(
    `${API_PREFIX}/collections/${encodeURIComponent(collection)}/chunks/${encodeURIComponent(chunkId)}`,
  );
  return result.data;
}
