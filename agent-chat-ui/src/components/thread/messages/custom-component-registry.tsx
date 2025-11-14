/**
 * 自定义组件注册表
 * 根据消息的 metadata 或 tool_calls 自动渲染前端组件
 */

import { Message } from "@langchain/langgraph-sdk";
import dynamic from "next/dynamic";
import { useState, useRef, useEffect } from "react";
import {
  DownloadIcon,
  Maximize2Icon,
  ChevronLeftIcon,
  ChevronRightIcon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

// 动态导入真实的图表组件
const ChartComponent = dynamic(
  () =>
    import("@/components/ui/chart-component").then((mod) => mod.ChartComponent),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-[400px] items-center justify-center rounded-lg border bg-gray-50">
        <div className="text-center">
          <div className="mb-2 text-lg">📊</div>
          <div className="text-sm text-gray-600">加载图表中...</div>
        </div>
      </div>
    ),
  },
);

// 组件类型定义
export type CustomComponentType =
  | "chart"
  | "table"
  | "image"
  | "file"
  | "custom";

// 组件配置
export interface ComponentConfig {
  type: CustomComponentType;
  data: any;
  metadata?: Record<string, unknown>;
}

// 从消息中提取组件配置
export function extractComponentConfig(
  message: Message,
): ComponentConfig | null {
  // 方式1: 从 message.additional_kwargs 中提取
  const additionalKwargs = (message as any).additional_kwargs;

  // 检查是否有 component 字段
  if (additionalKwargs?.component) {
    return additionalKwargs.component as ComponentConfig;
  }

  // 检查是否直接包含 type 和 data 字段（echart_agent 格式）
  if (additionalKwargs?.type && additionalKwargs?.data) {
    return {
      type: additionalKwargs.type as CustomComponentType,
      data: additionalKwargs.data,
    } as ComponentConfig;
  }

  // 方式2: 从 tool_calls 中提取（如果有特定的可视化工具调用）
  if ("tool_calls" in message && message.tool_calls) {
    const visualizationTool = message.tool_calls.find(
      (tc) => tc.name === "render_chart" || tc.name === "create_visualization",
    );

    if (visualizationTool?.args) {
      return {
        type: "chart",
        data: visualizationTool.args,
        metadata: { tool_call_id: visualizationTool.id },
      };
    }
  }

  // 方式3: 从消息内容中解析特殊标记
  const content = typeof message.content === "string" ? message.content : "";

  // 检测是否包含图表数据标记
  const chartMatch = content.match(/```chart\n([\s\S]*?)\n```/);
  if (chartMatch) {
    try {
      const chartData = JSON.parse(chartMatch[1]);
      return {
        type: "chart",
        data: chartData,
      };
    } catch (e) {
      console.error("Failed to parse chart data:", e);
    }
  }

  // 检测是否包含表格数据标记
  const tableMatch = content.match(/```table\n([\s\S]*?)\n```/);
  if (tableMatch) {
    try {
      const tableData = JSON.parse(tableMatch[1]);
      return {
        type: "table",
        data: tableData,
      };
    } catch (e) {
      console.error("Failed to parse table data:", e);
    }
  }

  return null;
}

// 组件渲染器
export function CustomComponentRenderer({
  config,
}: {
  config: ComponentConfig;
}) {
  switch (config.type) {
    case "chart":
      return (
        <div className="my-4 w-full">
          <ChartComponent
            config={config.data}
            className="w-full rounded-lg border shadow-sm"
          />
        </div>
      );

    case "table":
      return (
        <div className="my-4 overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                {config.data.headers?.map((header: string, idx: number) => (
                  <th
                    key={idx}
                    className="px-6 py-3 text-left text-xs font-medium tracking-wider text-gray-500 uppercase"
                  >
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 bg-white">
              {config.data.rows?.map((row: any[], rowIdx: number) => (
                <tr key={rowIdx}>
                  {row.map((cell, cellIdx) => (
                    <td
                      key={cellIdx}
                      className="px-6 py-4 text-sm whitespace-nowrap text-gray-900"
                    >
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );

    case "image": {
      return <ImageComponent config={config} />;
    }

    case "file":
      return (
        <div className="my-4 flex items-center gap-2 rounded-lg border p-4">
          <svg
            className="h-6 w-6 text-gray-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
            />
          </svg>
          <div className="flex-1">
            <p className="text-sm font-medium">{config.data.name}</p>
            {config.data.size && (
              <p className="text-xs text-gray-500">
                {formatFileSize(config.data.size)}
              </p>
            )}
          </div>
          {config.data.url && (
            <a
              href={config.data.url}
              download={config.data.name}
              className="text-blue-600 hover:text-blue-800"
            >
              Download
            </a>
          )}
        </div>
      );

    default:
      return (
        <div className="my-4 rounded-lg border border-yellow-200 bg-yellow-50 p-4">
          <p className="text-sm text-yellow-800">
            Unknown component type: {config.type}
          </p>
          <pre className="mt-2 overflow-auto text-xs text-gray-600">
            {JSON.stringify(config.data, null, 2)}
          </pre>
        </div>
      );
  }
}

// 辅助函数：格式化文件大小
function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

// 图片卡片组件（带悬停操作按钮）
function ImageCard({
  url,
  alt,
  caption,
  onPreview,
  onDownload,
}: {
  url: string;
  alt?: string;
  caption?: string;
  onPreview: () => void;
  onDownload: () => void;
}) {
  const [isHovering, setIsHovering] = useState(false);

  return (
    <div
      className="relative flex flex-col"
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
    >
      <div className="relative overflow-hidden rounded-lg shadow-md">
        <img
          src={url}
          alt={alt || "Image"}
          className="h-full w-full object-cover"
        />
        {/* 悬停时显示的操作按钮 */}
        <div
          className="absolute top-2 right-2 flex gap-1 transition-opacity"
          style={{ opacity: isHovering ? 1 : 0 }}
        >
          <Button
            variant="ghost"
            size="icon"
            className="text-foreground size-8 bg-white/95 shadow-sm hover:bg-white"
            onClick={onDownload}
            aria-label="下载图片"
          >
            <DownloadIcon className="size-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="text-foreground size-8 bg-white/95 shadow-sm hover:bg-white"
            onClick={onPreview}
            aria-label="放大查看"
          >
            <Maximize2Icon className="size-4" />
          </Button>
        </div>
      </div>
      {caption && (
        <p className="mt-2 text-center text-sm text-gray-600">{caption}</p>
      )}
    </div>
  );
}

// 图片组件（支持单张/多张、网格/轮播）
function ImageComponent({ config }: { config: ComponentConfig }) {
  const [activeImage, setActiveImage] = useState<{
    url: string;
    alt?: string;
    caption?: string;
  } | null>(null);
  const [isDialogOpen, setDialogOpen] = useState(false);

  // 支持单张或多张图片
  const images = Array.isArray(config.data.images)
    ? config.data.images
    : config.data.url
      ? [
          {
            url: config.data.url,
            alt: config.data.alt,
            caption: config.data.caption,
          },
        ]
      : [];

  if (images.length === 0) return null;

  const handlePreview = (img: any) => {
    setActiveImage(img);
    setDialogOpen(true);
  };

  const handleDownload = (img: any) => {
    const link = document.createElement("a");
    link.href = img.url;
    link.download = img.alt || "image";
    link.rel = "noopener";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // 单张图片
  if (images.length === 1) {
    return (
      <>
        <div className="my-4">
          <ImageCard
            url={images[0].url}
            alt={images[0].alt}
            caption={images[0].caption}
            onPreview={() => handlePreview(images[0])}
            onDownload={() => handleDownload(images[0])}
          />
        </div>
        <Dialog
          open={isDialogOpen}
          onOpenChange={setDialogOpen}
        >
          <DialogContent className="max-w-5xl">
            <DialogHeader>
              <DialogTitle>{activeImage?.alt || "图片预览"}</DialogTitle>
            </DialogHeader>
            {activeImage && (
              <div className="flex flex-col gap-4">
                <div className="bg-muted/20 flex max-h-[75vh] items-center justify-center overflow-auto rounded-md border p-4">
                  <img
                    src={activeImage.url}
                    alt={activeImage.alt || "放大图片"}
                    className="max-w-full rounded-sm object-contain"
                  />
                </div>
                {activeImage.caption && (
                  <p className="text-muted-foreground text-sm leading-relaxed">
                    {activeImage.caption}
                  </p>
                )}
                <div className="flex items-center justify-end">
                  <Button onClick={() => handleDownload(activeImage)}>
                    下载
                  </Button>
                </div>
              </div>
            )}
          </DialogContent>
        </Dialog>
      </>
    );
  }

  // 多张图片 - 网格布局
  const layout = config.data.layout || "grid";

  if (layout === "grid") {
    return (
      <>
        <div className="my-4">
          <div
            className={`grid gap-4 ${
              images.length === 2
                ? "grid-cols-2"
                : images.length === 3
                  ? "grid-cols-3"
                  : images.length === 4
                    ? "grid-cols-2"
                    : "grid-cols-3"
            }`}
          >
            {images.map((img: any, idx: number) => (
              <ImageCard
                key={idx}
                url={img.url}
                alt={img.alt || `Image ${idx + 1}`}
                caption={img.caption}
                onPreview={() => handlePreview(img)}
                onDownload={() => handleDownload(img)}
              />
            ))}
          </div>
          {config.data.caption && (
            <p className="mt-4 text-center text-sm text-gray-600">
              {config.data.caption}
            </p>
          )}
        </div>
        <Dialog
          open={isDialogOpen}
          onOpenChange={setDialogOpen}
        >
          <DialogContent className="max-w-5xl">
            <DialogHeader>
              <DialogTitle>{activeImage?.alt || "图片预览"}</DialogTitle>
            </DialogHeader>
            {activeImage && (
              <div className="flex flex-col gap-4">
                <div className="bg-muted/20 flex max-h-[75vh] items-center justify-center overflow-auto rounded-md border p-4">
                  <img
                    src={activeImage.url}
                    alt={activeImage.alt || "放大图片"}
                    className="max-w-full rounded-sm object-contain"
                  />
                </div>
                {activeImage.caption && (
                  <p className="text-muted-foreground text-sm leading-relaxed">
                    {activeImage.caption}
                  </p>
                )}
                <div className="flex items-center justify-end">
                  <Button onClick={() => handleDownload(activeImage)}>
                    下载
                  </Button>
                </div>
              </div>
            )}
          </DialogContent>
        </Dialog>
      </>
    );
  }

  // 轮播图布局
  return (
    <CarouselComponent
      images={images}
      caption={config.data.caption}
    />
  );
}

// 轮播图组件（带导航按钮和指示器）
function CarouselComponent({
  images,
  caption,
}: {
  images: any[];
  caption?: string;
}) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isHovering, setIsHovering] = useState(false);
  const [activeImage, setActiveImage] = useState<any>(null);
  const [isDialogOpen, setDialogOpen] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const handlePreview = (img: any) => {
    setActiveImage(img);
    setDialogOpen(true);
  };

  const handleDownload = (img: any) => {
    const link = document.createElement("a");
    link.href = img.url;
    link.download = img.alt || "image";
    link.rel = "noopener";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const scrollToIndex = (index: number) => {
    if (scrollRef.current) {
      const scrollWidth = scrollRef.current.scrollWidth;
      const targetScroll = (scrollWidth / images.length) * index;
      scrollRef.current.scrollTo({
        left: targetScroll,
        behavior: "smooth",
      });
      setCurrentIndex(index);
    }
  };

  const handlePrev = () => {
    const newIndex = currentIndex > 0 ? currentIndex - 1 : images.length - 1;
    scrollToIndex(newIndex);
  };

  const handleNext = () => {
    const newIndex = currentIndex < images.length - 1 ? currentIndex + 1 : 0;
    scrollToIndex(newIndex);
  };

  // 监听滚动更新当前索引
  useEffect(() => {
    const handleScroll = () => {
      if (scrollRef.current) {
        const scrollLeft = scrollRef.current.scrollLeft;
        const width = scrollRef.current.clientWidth;
        const newIndex = Math.round(scrollLeft / width);
        setCurrentIndex(newIndex);
      }
    };

    const scrollElement = scrollRef.current;
    if (scrollElement) {
      scrollElement.addEventListener("scroll", handleScroll);
      return () => scrollElement.removeEventListener("scroll", handleScroll);
    }
  }, []);

  return (
    <>
      <div className="my-4">
        <div
          className="relative mx-auto max-w-3xl"
          onMouseEnter={() => setIsHovering(true)}
          onMouseLeave={() => setIsHovering(false)}
        >
          {/* 轮播容器 */}
          <div
            ref={scrollRef}
            className="scrollbar-hide flex snap-x snap-mandatory overflow-x-auto rounded-lg shadow-md"
            style={{ scrollbarWidth: "none", msOverflowStyle: "none" }}
          >
            {images.map((img: any, idx: number) => (
              <div
                key={idx}
                className="relative min-w-full snap-center bg-gray-100"
              >
                <img
                  src={img.url}
                  alt={img.alt || `Image ${idx + 1}`}
                  className="max-h-[500px] w-full object-contain"
                />
                {/* 悬停时显示的操作按钮 */}
                <div
                  className="absolute top-2 right-2 flex gap-1 transition-opacity"
                  style={{ opacity: isHovering ? 1 : 0 }}
                >
                  <Button
                    variant="ghost"
                    size="icon"
                    className="text-foreground size-8 bg-white/95 shadow-sm hover:bg-white"
                    onClick={() => handleDownload(img)}
                    aria-label="下载图片"
                  >
                    <DownloadIcon className="size-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="text-foreground size-8 bg-white/95 shadow-sm hover:bg-white"
                    onClick={() => handlePreview(img)}
                    aria-label="放大查看"
                  >
                    <Maximize2Icon className="size-4" />
                  </Button>
                </div>
                {/* 图片标题 */}
                {img.caption && (
                  <div className="absolute right-0 bottom-0 left-0 bg-gradient-to-t from-black/70 to-transparent p-3">
                    <p className="text-center text-sm text-white">
                      {img.caption}
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* 左右导航按钮 */}
          {images.length > 1 && (
            <>
              <Button
                variant="ghost"
                size="icon"
                className="text-foreground absolute top-1/2 left-2 size-10 -translate-y-1/2 bg-white/90 shadow-md hover:bg-white"
                onClick={handlePrev}
                aria-label="上一张"
                style={{ opacity: isHovering ? 1 : 0.6 }}
              >
                <ChevronLeftIcon className="size-5" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="text-foreground absolute top-1/2 right-2 size-10 -translate-y-1/2 bg-white/90 shadow-md hover:bg-white"
                onClick={handleNext}
                aria-label="下一张"
                style={{ opacity: isHovering ? 1 : 0.6 }}
              >
                <ChevronRightIcon className="size-5" />
              </Button>
            </>
          )}

          {/* 指示器 */}
          {images.length > 1 && (
            <div className="absolute right-0 bottom-4 left-0 flex justify-center gap-2">
              {images.map((_, idx) => (
                <button
                  key={idx}
                  className="size-2 rounded-full transition-all"
                  style={{
                    backgroundColor:
                      idx === currentIndex
                        ? "rgba(255, 255, 255, 0.9)"
                        : "rgba(255, 255, 255, 0.5)",
                    width: idx === currentIndex ? "24px" : "8px",
                  }}
                  onClick={() => scrollToIndex(idx)}
                  aria-label={`跳转到第 ${idx + 1} 张`}
                />
              ))}
            </div>
          )}
        </div>

        {/* 整体说明和计数 */}
        <div className="mt-2 flex items-center justify-center gap-2 text-sm text-gray-600">
          {images.length > 1 && (
            <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs">
              {currentIndex + 1} / {images.length}
            </span>
          )}
          {caption && <span>{caption}</span>}
        </div>
      </div>

      {/* 预览对话框 */}
      <Dialog
        open={isDialogOpen}
        onOpenChange={setDialogOpen}
      >
        <DialogContent className="max-w-5xl">
          <DialogHeader>
            <DialogTitle>{activeImage?.alt || "图片预览"}</DialogTitle>
          </DialogHeader>
          {activeImage && (
            <div className="flex flex-col gap-4">
              <div className="bg-muted/20 flex max-h-[75vh] items-center justify-center overflow-auto rounded-md border p-4">
                <img
                  src={activeImage.url}
                  alt={activeImage.alt || "放大图片"}
                  className="max-w-full rounded-sm object-contain"
                />
              </div>
              {activeImage.caption && (
                <p className="text-muted-foreground text-sm leading-relaxed">
                  {activeImage.caption}
                </p>
              )}
              <div className="flex items-center justify-end">
                <Button onClick={() => handleDownload(activeImage)}>
                  下载
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
