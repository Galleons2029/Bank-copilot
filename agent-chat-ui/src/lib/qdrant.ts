const DEFAULT_QDRANT_URL = process.env.QDRANT_URL ?? "http://127.0.0.1:6333";

export const KNOWLEDGE_METADATA_POINT_ID = "__knowledge_metadata__";

type VectorParams =
  | number
  | {
      size: number;
      distance?: string;
    }
  | Record<string, { size: number; distance?: string }>;

type CollectionParams = {
  vectors: VectorParams;
};

type CollectionConfig = {
  params: CollectionParams;
};

export type QdrantCollection = {
  name: string;
  status: string;
};

export type QdrantCollectionDetail = {
  status: string;
  optimizer_status: string;
  vectors_count: number;
  points_count: number;
  config: CollectionConfig;
};

type QdrantResponse<T> = {
  result: T;
};

export async function qdrantRequest<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const url =
    path.startsWith("http://") || path.startsWith("https://")
      ? path
      : `${DEFAULT_QDRANT_URL}${path}`;

  const response = await fetch(url, {
    ...init,
    headers: {
      "content-type": "application/json",
      ...(init.headers ?? {}),
    },
    cache: "no-store",
  });

  if (!response.ok) {
    const message = await safeParseError(response);
    throw new Error(
      `Qdrant request failed (${response.status}): ${message ?? response.statusText}`,
    );
  }

  if (response.status === 204) {
    return {} as T;
  }

  return (await response.json()) as T;
}

export function resolveVectorSize(config: CollectionConfig): number {
  const { vectors } = config.params;

  if (typeof vectors === "number") {
    return vectors;
  }

  if ("size" in vectors) {
    return vectors.size;
  }

  const [first] = Object.values(vectors);
  if (first?.size) {
    return first.size;
  }

  throw new Error("无法确定集合的向量维度");
}

export function buildZeroVector(dimension: number): number[] {
  if (dimension <= 0) {
    throw new Error("向量维度必须为正数");
  }

  return Array.from({ length: dimension }, () => 0);
}

export function generateDeterministicVector(
  seed: string,
  dimension: number,
): number[] {
  if (dimension <= 0) {
    throw new Error("向量维度必须为正数");
  }

  const vector = Array.from({ length: dimension }, () => 0);

  if (!seed) {
    return vector;
  }

  for (let i = 0; i < seed.length; i++) {
    const index = i % dimension;
    const charCode = seed.charCodeAt(i);
    vector[index] += (charCode % 97) / 97;
  }

  const total = vector.reduce((sum, value) => sum + value, 0);
  if (total === 0) {
    return vector;
  }

  return vector.map((value) => value / total);
}

async function safeParseError(response: Response) {
  try {
    const json = await response.json();
    if (typeof json === "string") {
      return json;
    }
    if (json?.status?.error) {
      return json.status.error;
    }
    if (json?.error) {
      return json.error;
    }
    return JSON.stringify(json);
  } catch {
    return null;
  }
}
