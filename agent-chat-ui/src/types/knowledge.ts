export type KnowledgeMetadata = {
  displayName: string;
  description?: string;
  tags?: string[];
};

export type KnowledgeBase = {
  name: string;
  status: string;
  vectorSize: number;
  distance?: string;
  chunkCount: number;
  metadata: KnowledgeMetadata;
};

export type KnowledgeChunk = {
  id: string;
  text: string;
  title?: string;
  source?: string;
  tags?: string[];
  metadata?: Record<string, unknown>;
  updatedAt?: string;
};
