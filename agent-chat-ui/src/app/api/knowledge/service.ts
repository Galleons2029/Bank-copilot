import { randomUUID } from "crypto";

import {
  KNOWLEDGE_METADATA_POINT_ID,
  buildZeroVector,
  generateDeterministicVector,
  qdrantRequest,
  resolveVectorSize,
  type QdrantCollection,
  type QdrantCollectionDetail,
} from "@/lib/qdrant";
import {
  type KnowledgeBase,
  type KnowledgeChunk,
  type KnowledgeMetadata,
} from "@/types/knowledge";

type QdrantResponse<T> = {
  result: T;
};

type ScrollResult = {
  points: QdrantPoint[];
  next_page_offset?: string | number;
};

type QdrantPoint = {
  id: string | number;
  payload?: ChunkPayload;
};

type ChunkPayload = {
  kind?: "chunk" | "metadata";
  text?: string;
  title?: string;
  source?: string;
  tags?: string[];
  metadata?: Record<string, unknown>;
  updatedAt?: string;
  displayName?: string;
  description?: string;
};

const METADATA_FILTER = {
  must: [
    {
      key: "kind",
      match: {
        value: "metadata",
      },
    },
  ],
};

const CHUNK_FILTER = {
  must_not: [
    {
      key: "kind",
      match: {
        value: "metadata",
      },
    },
  ],
};

export async function listKnowledgeBases(): Promise<KnowledgeBase[]> {
  const { result } = await qdrantRequest<QdrantResponse<{ collections: QdrantCollection[] }>>(
    "/collections",
  );

  if (result.collections.length === 0) {
    return [];
  }

  const bases = await Promise.all(
    result.collections.map(({ name }) => getKnowledgeBase(name)),
  );

  return bases.sort((a, b) =>
    a.metadata.displayName.localeCompare(b.metadata.displayName, "zh-CN"),
  );
}

export async function getKnowledgeBase(name: string): Promise<KnowledgeBase> {
  const detail = await getCollectionDetail(name);
  const vectorSize = resolveVectorSize(detail.config);
  const distance = resolveDistance(detail.config);
  const metadata = await fetchMetadata(name);
  const chunkCount = Math.max(
    detail.points_count - (metadata ? 1 : 0),
    0,
  );

  return {
    name,
    status: detail.status,
    vectorSize,
    distance,
    chunkCount,
    metadata: {
      displayName: metadata?.displayName ?? name,
      description: metadata?.description,
      tags: metadata?.tags ?? [],
    },
  };
}

export async function createKnowledgeBase(input: {
  name: string;
  vectorSize: number;
  distance?: string;
  description?: string;
  displayName?: string;
}): Promise<KnowledgeBase> {
  const normalizedName = input.name.trim();
  const vectorConfig = {
    vectors: {
      size: input.vectorSize,
      distance: input.distance ?? "Cosine",
    },
  };

  await qdrantRequest(`/collections/${encodeURIComponent(normalizedName)}`, {
    method: "PUT",
    body: JSON.stringify(vectorConfig),
  });

  await upsertMetadata(normalizedName, {
    displayName: input.displayName?.trim() || normalizedName,
    description: input.description?.trim(),
  });

  return getKnowledgeBase(normalizedName);
}

export async function deleteKnowledgeBase(name: string) {
  await qdrantRequest(`/collections/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}

export async function upsertMetadata(
  collection: string,
  metadata: Partial<KnowledgeMetadata>,
) {
  const detail = await getCollectionDetail(collection);
  const vectorSize = resolveVectorSize(detail.config);

  await qdrantRequest(`/collections/${encodeURIComponent(collection)}/points`, {
    method: "PUT",
    body: JSON.stringify({
      points: [
        {
          id: KNOWLEDGE_METADATA_POINT_ID,
          vector: buildZeroVector(vectorSize),
          payload: {
            kind: "metadata",
            displayName: metadata.displayName ?? collection,
            description: metadata.description,
            tags: metadata.tags ?? [],
          },
        },
      ],
    }),
  });
}

export async function fetchChunks(
  collection: string,
  options: { limit?: number; offset?: string | number } = {},
) {
  const { limit = 50, offset } = options;

  const { result } = await qdrantRequest<QdrantResponse<ScrollResult>>(
    `/collections/${encodeURIComponent(collection)}/points/scroll`,
    {
      method: "POST",
      body: JSON.stringify({
        limit,
        offset,
        with_payload: true,
        with_vectors: false,
        filter: CHUNK_FILTER,
      }),
    },
  );

  return {
    chunks: result.points.map(formatChunk),
    nextOffset: result.next_page_offset,
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
  const detail = await getCollectionDetail(collection);
  const vectorSize = resolveVectorSize(detail.config);

  const pointId = input.id ?? randomUUID();
  const payload: ChunkPayload = {
    kind: "chunk",
    text: input.text,
    title: input.title,
    source: input.source,
    tags: input.tags ?? [],
    metadata: input.metadata ?? {},
    updatedAt: new Date().toISOString(),
  };

  await qdrantRequest(`/collections/${encodeURIComponent(collection)}/points`, {
    method: "PUT",
    body: JSON.stringify({
      points: [
        {
          id: pointId,
          payload,
          vector: generateDeterministicVector(
            [input.title, input.text, pointId].filter(Boolean).join(" "),
            vectorSize,
          ),
        },
      ],
    }),
  });

  return {
    id: pointId,
    text: payload.text ?? "",
    title: payload.title,
    source: payload.source,
    tags: payload.tags ?? [],
    metadata: payload.metadata,
    updatedAt: payload.updatedAt,
  };
}

export async function deleteChunk(collection: string, chunkId: string) {
  await qdrantRequest(
    `/collections/${encodeURIComponent(collection)}/points/delete`,
    {
      method: "POST",
      body: JSON.stringify({
        points: [chunkId],
      }),
    },
  );
}

export async function getChunk(collection: string, chunkId: string) {
  const { result } = await qdrantRequest<
    QdrantResponse<{ points: QdrantPoint[] }>
  >(`/collections/${encodeURIComponent(collection)}/points/retrieve`, {
    method: "POST",
    body: JSON.stringify({
      ids: [chunkId],
      with_payload: true,
      with_vectors: false,
    }),
  });

  const [point] = result.points;
  if (!point) {
    return null;
  }

  return formatChunk(point);
}

async function fetchMetadata(
  collection: string,
): Promise<KnowledgeMetadata | null> {
  const { result } = await qdrantRequest<QdrantResponse<ScrollResult>>(
    `/collections/${encodeURIComponent(collection)}/points/scroll`,
    {
      method: "POST",
      body: JSON.stringify({
        limit: 1,
        with_payload: true,
        with_vectors: false,
        filter: METADATA_FILTER,
      }),
    },
  );

  const [point] = result.points;
  if (!point?.payload) {
    return null;
  }

  return {
    displayName:
      point.payload.displayName ?? point.payload.metadata?.displayName ?? "",
    description:
      point.payload.description ?? point.payload.metadata?.description,
    tags: (point.payload.tags ?? []) as string[],
  };
}

function resolveDistance(config: QdrantCollectionDetail["config"]) {
  const vectors = config.params.vectors;

  if (!vectors) {
    return undefined;
  }

  if (typeof vectors === "number") {
    return undefined;
  }

  if ("distance" in vectors) {
    return vectors.distance;
  }

  const values = Object.values(vectors);
  return values[0]?.distance;
}

async function getCollectionDetail(name: string) {
  const { result } = await qdrantRequest<QdrantResponse<QdrantCollectionDetail>>(
    `/collections/${encodeURIComponent(name)}`,
  );

  return result;
}

function formatChunk(point: QdrantPoint): KnowledgeChunk {
  const payload = point.payload ?? {};

  return {
    id: String(point.id),
    text: payload.text ?? "",
    title: payload.title,
    source: payload.source,
    tags: (payload.tags ?? []) as string[],
    metadata: payload.metadata,
    updatedAt: payload.updatedAt,
  };
}
