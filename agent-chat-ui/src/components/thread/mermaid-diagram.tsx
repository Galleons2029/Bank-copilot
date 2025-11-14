"use client";

import { useEffect, useRef, useState } from "react";
import mermaid from "mermaid";
import { ChevronDown, ChevronUp, Code2 } from "lucide-react";

interface MermaidDiagramProps {
  chart: string;
  className?: string;
}

// 初始化 mermaid 配置
let mermaidInitialized = false;

export function MermaidDiagram({ chart, className = "" }: MermaidDiagramProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [svg, setSvg] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [showSource, setShowSource] = useState(false);
  const renderIdRef = useRef(0);

  useEffect(() => {
    // 初始化 mermaid（只需要一次）
    if (!mermaidInitialized) {
      mermaid.initialize({
        startOnLoad: false,
        theme: "default",
        securityLevel: "loose",
        fontFamily: "inherit",
        flowchart: {
          useMaxWidth: false,
          wrappingWidth: 660, // double the previous width so the rendered blocks look twice as large
        },
        // 设置透明背景
        themeVariables: {
          background: "transparent",

        },
      });
      mermaidInitialized = true;
    }
  }, []);

  useEffect(() => {
    const renderDiagram = async () => {
      if (!chart || !containerRef.current) return;

      try {
        setError(null);

        // 生成唯一 ID
        renderIdRef.current += 1;
        const id = `mermaid-diagram-${Date.now()}-${renderIdRef.current}`;

        // 渲染图表
        const { svg: renderedSvg } = await mermaid.render(id, chart);
        setSvg(renderedSvg);
      } catch (err) {
        console.error("Mermaid rendering error:", err);
        setError(err instanceof Error ? err.message : String(err));
      }
    };

    renderDiagram();
  }, [chart]);

  if (error) {
    return (
      <div className="my-4 rounded-lg border border-red-200 bg-red-50 p-4">
        <p className="mb-2 font-semibold text-red-700">
          ⚠️ Mermaid 图表渲染错误
        </p>
        <pre className="overflow-auto text-xs text-red-600">{error}</pre>
        <details className="mt-2">
          <summary className="cursor-pointer text-sm text-red-600 hover:text-red-800">
            查看原始代码
          </summary>
          <pre className="mt-2 overflow-auto rounded bg-red-100 p-2 text-xs">
            {chart}
          </pre>
        </details>
      </div>
    );
  }

  return (
    <div
      className="my-4 rounded-lg p-0"
      style={{ backgroundColor: "transparent" }}
    >
      {/* 折叠按钮 */}
      <div className="mb-2 flex items-center justify-end">
        <button
          onClick={() => setShowSource(!showSource)}
          className="flex items-center gap-1 rounded-md border border-gray-300 bg-white px-3 py-1.5 text-sm text-gray-700 transition-colors hover:border-gray-400 hover:bg-gray-50"
          title={showSource ? "隐藏源代码" : "查看源代码"}
        >
          <Code2 size={14} />
          <span>{showSource ? "隐藏" : "查看"}源代码</span>
          {showSource ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
      </div>

      {/* 源代码展示 */}
      {showSource && (
        <div className="mb-3 overflow-auto rounded-lg border border-gray-300 bg-gray-50 p-3">
          <div className="mb-1 flex items-center gap-2 text-xs font-medium text-gray-600">
            <Code2 size={12} />
            <span>Mermaid 源代码</span>
          </div>
          <pre className="overflow-auto text-xs text-gray-800">
            <code>{chart}</code>
          </pre>
        </div>
      )}

      {/* 图表渲染区域 */}
      <div
        ref={containerRef}
        className={`flex justify-center overflow-auto rounded-lg p-4 ${className}`}
        style={{ backgroundColor: "transparent" }}
        dangerouslySetInnerHTML={{ __html: svg }}
      />
    </div>
  );
}
