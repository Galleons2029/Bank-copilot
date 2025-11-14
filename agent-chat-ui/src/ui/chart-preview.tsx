"use client";

import * as React from "react";
import { DownloadIcon, Maximize2Icon } from "lucide-react";
import "./style.css";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

import type { ChartImage, ChartPreviewProps } from "./chart-preview.types";
import { getChartFilename, toDataUrl } from "./chart-preview.types";

interface PreparedChart extends ChartImage {
  url: string;
}

function prepareCharts(charts?: ChartImage[]): PreparedChart[] {
  if (!charts?.length) {
    return [];
  }

  return charts
    .map((chart) => ({
      ...chart,
      url: toDataUrl(chart),
    }))
    .filter((chart): chart is PreparedChart => Boolean(chart.url));
}

interface ChartCardProps {
  chart: PreparedChart;
  onDownload: (chart: PreparedChart) => void;
  onPreview: (chart: PreparedChart) => void;
}

function ChartCard({ chart, onDownload, onPreview }: ChartCardProps) {
  const [isHovering, setIsHovering] = React.useState(false);
  const containerRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleMouseEnter = () => setIsHovering(true);
    const handleMouseLeave = () => setIsHovering(false);

    container.addEventListener("mouseenter", handleMouseEnter);
    container.addEventListener("mouseleave", handleMouseLeave);

    return () => {
      container.removeEventListener("mouseenter", handleMouseEnter);
      container.removeEventListener("mouseleave", handleMouseLeave);
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative flex-shrink-0"
    >
      <div
        className="relative h-[180px] w-[280px]"
        style={{ height: "180px", width: "280px" }}
      >
        <div className="bg-muted/10 relative flex h-full w-full items-center justify-center overflow-hidden rounded-md">
          <img
            src={chart.url}
            alt={chart.title ?? "生成的图表"}
            className="pointer-events-none max-h-[170px] max-w-full object-contain"
            style={{ maxHeight: "170px" }}
          />
          <div
            className="absolute top-2 right-2 flex gap-1"
            style={{
              pointerEvents: "auto",
              opacity: isHovering ? 1 : 0,
              transition: "opacity 200ms",
            }}
          >
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-foreground size-8 shadow-sm"
                  style={{ backgroundColor: "rgba(255, 255, 255, 0.95)" }}
                  onClick={() => onDownload(chart)}
                  aria-label="下载"
                >
                  <DownloadIcon className="size-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>下载</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-foreground size-8 shadow-sm"
                  style={{ backgroundColor: "rgba(255, 255, 255, 0.95)" }}
                  onClick={() => onPreview(chart)}
                  aria-label="放大查看图表"
                >
                  <Maximize2Icon className="size-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>预览</TooltipContent>
            </Tooltip>
          </div>
        </div>
      </div>
    </div>
  );
}

export function ChartPreview({ charts }: ChartPreviewProps): React.ReactNode {
  const [activeChart, setActiveChart] = React.useState<PreparedChart | null>(
    null,
  );
  const [isDialogOpen, setDialogOpen] = React.useState(false);

  const preparedCharts = React.useMemo(() => prepareCharts(charts), [charts]);

  const handlePreview = React.useCallback((chart: PreparedChart) => {
    setActiveChart(chart);
    setDialogOpen(true);
  }, []);

  const handleDownload = React.useCallback((chart: PreparedChart) => {
    if (typeof window === "undefined") {
      return;
    }

    const link = document.createElement("a");
    link.href = chart.url;
    link.download = getChartFilename(chart);
    link.rel = "noopener";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, []);

  const hasCharts = preparedCharts.length > 0;

  if (!hasCharts) {
    return null;
  }

  return (
    <section className="w-full">
      <div className="flex gap-3 overflow-x-auto pb-2">
        {preparedCharts.map((chart) => (
          <ChartCard
            key={chart.id}
            chart={chart}
            onDownload={handleDownload}
            onPreview={handlePreview}
          />
        ))}
      </div>

      <Dialog
        open={isDialogOpen}
        onOpenChange={setDialogOpen}
      >
        <DialogContent className="max-w-5xl">
          <DialogHeader>
            <DialogTitle>{activeChart?.title ?? "图表预览"}</DialogTitle>
          </DialogHeader>
          {activeChart ? (
            <div className="flex flex-col gap-4">
              <div className="bg-muted/20 flex max-h-[75vh] items-center justify-center overflow-auto rounded-md border p-4">
                <img
                  src={activeChart.url}
                  alt={activeChart.title ?? "放大图表"}
                  className="max-w-full rounded-sm object-contain"
                />
              </div>
              {activeChart.description ? (
                <p className="text-muted-foreground text-sm leading-relaxed">
                  {activeChart.description}
                </p>
              ) : null}
              <div className="flex items-center justify-end">
                <Button onClick={() => handleDownload(activeChart)}>
                  下载
                </Button>
              </div>
            </div>
          ) : (
            <div className="text-muted-foreground flex items-center justify-center py-8">
              暂无选中的图表
            </div>
          )}
        </DialogContent>
      </Dialog>
    </section>
  );
}

export type { ChartPreviewProps, ChartImage };
export default {
  "chart-preview": ChartPreview,
};
