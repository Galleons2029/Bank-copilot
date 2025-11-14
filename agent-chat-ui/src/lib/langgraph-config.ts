import { getApiKey } from "@/lib/api-key";

export const DEFAULT_API_URL = "http://localhost:2024";
export const DEFAULT_ASSISTANT_ID = "agent";

export function resolveApiUrl(): string {
  return process.env.NEXT_PUBLIC_API_URL || DEFAULT_API_URL;
}

export function resolveAssistantId(): string {
  return process.env.NEXT_PUBLIC_ASSISTANT_ID || DEFAULT_ASSISTANT_ID;
}

export function resolveApiKey(): string {
  return process.env.NEXT_PUBLIC_API_KEY || getApiKey() || "";
}
