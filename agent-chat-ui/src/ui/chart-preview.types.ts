// {
//     "type": "image",
//     "data": chart["data"],
//     "source_type": "base64",
//     "mime_type": chart["mimeType"],
// }
export interface ChartImage {
  id?: string;
  title?: string;
  description?: string;
  type?: string;
  data?: string;
  mimeType?: string;
  mime_type?: string;
  source_type?: string;
}

export interface ChartPreviewProps {
  charts?: ChartImage[];
}

function resolveMimeType(chart: ChartImage): string {
  const fallback =
    chart.type === "image" ? "image/png" : "application/octet-stream";
  return chart.mimeType || chart.mime_type || fallback;
}

function cleanData(data: string | undefined): string {
  if (!data) {
    return "";
  }

  const trimmed = data.trim();
  if (trimmed.startsWith("data:")) {
    return trimmed;
  }

  return trimmed.replace(/\s+/g, "");
}

export function toDataUrl(chart: ChartImage): string {
  const cleaned = cleanData(chart.data);
  if (!cleaned) {
    return "";
  }

  if (cleaned.startsWith("data:")) {
    return cleaned;
  }

  if (chart.source_type && chart.source_type !== "base64") {
    return "";
  }

  const mime = resolveMimeType(chart);
  return `data:${mime};base64,${cleaned}`;
}

function resolveExtension(mime: string): string {
  const [, ext] = mime.split("/");
  return ext || "png";
}

function sanitizeFilenameSegment(segment: string): string {
  return (
    segment
      .trim()
      .replace(/[^a-zA-Z0-9._-]+/g, "-")
      .replace(/-+/g, "-")
      .replace(/^-|-$/g, "") || "chart"
  );
}

export function getChartFilename(
  chart: ChartImage,
  fallbackIndex?: number,
): string {
  const mime = resolveMimeType(chart);
  const extension = resolveExtension(mime);

  const baseNameSource =
    chart.title ||
    chart.id ||
    (fallbackIndex != null ? `chart-${fallbackIndex + 1}` : "chart");
  const baseName = sanitizeFilenameSegment(baseNameSource);

  return `${baseName}.${extension}`;
}
