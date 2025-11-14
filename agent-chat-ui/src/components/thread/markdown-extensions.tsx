/**
 * Markdown 扩展 - 自定义语法支持
 *
 * 支持在 Markdown 中使用自定义语法触发组件
 * 例如：:::chart{配置} 或 :::table{数据}
 */

import { ComponentType, ReactNode } from "react";
import dynamic from "next/dynamic";

// 懒加载图表组件
const ChartComponent = dynamic(
  () =>
    import("@/components/ui/chart-component").then((mod) => mod.ChartComponent),
  {
    ssr: false,
    loading: () => (
      <div className="h-64 animate-pulse rounded bg-gray-100"></div>
    ),
  },
);

/**
 * 自定义 Markdown 组件映射
 *
 * 这些组件会在 react-markdown 的 components prop 中使用
 */

interface ChartDirectiveProps {
  config?: string; // JSON 字符串
  type?: string; // 图表类型
  children?: ReactNode;
}

// 图表指令组件
export function ChartDirective({
  config,
  type,
  children,
}: ChartDirectiveProps) {
  try {
    const chartConfig = config ? JSON.parse(config) : null;

    if (!chartConfig) {
      return (
        <div className="rounded border border-red-200 bg-red-50 p-4">
          <p className="text-red-600">无效的图表配置</p>
        </div>
      );
    }

    return (
      <div className="my-4">
        <ChartComponent config={{ option: chartConfig }} />
      </div>
    );
  } catch (error) {
    return (
      <div className="rounded border border-red-200 bg-red-50 p-4">
        <p className="text-red-600">解析图表配置失败: {String(error)}</p>
      </div>
    );
  }
}

// 表格指令组件
interface TableDirectiveProps {
  data?: string;
  children?: ReactNode;
}

export function TableDirective({ data, children }: TableDirectiveProps) {
  try {
    const tableData = data ? JSON.parse(data) : null;

    if (!tableData?.headers || !tableData?.rows) {
      return (
        <div className="rounded border border-red-200 bg-red-50 p-4">
          <p className="text-red-600">无效的表格数据</p>
        </div>
      );
    }

    return (
      <div className="my-4 overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 rounded-lg border">
          <thead className="bg-gray-50">
            <tr>
              {tableData.headers.map((header: string, idx: number) => (
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
            {tableData.rows.map((row: any[], rowIdx: number) => (
              <tr
                key={rowIdx}
                className="hover:bg-gray-50"
              >
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
  } catch (error) {
    return (
      <div className="rounded border border-red-200 bg-red-50 p-4">
        <p className="text-red-600">解析表格数据失败: {String(error)}</p>
      </div>
    );
  }
}

// 高亮盒子指令
interface CalloutProps {
  type?: "info" | "warning" | "error" | "success";
  title?: string;
  children?: ReactNode;
}

export function Callout({ type = "info", title, children }: CalloutProps) {
  const styles = {
    info: "border-blue-200 bg-blue-50 text-blue-800",
    warning: "border-yellow-200 bg-yellow-50 text-yellow-800",
    error: "border-red-200 bg-red-50 text-red-800",
    success: "border-green-200 bg-green-50 text-green-800",
  };

  const icons = {
    info: "ℹ️",
    warning: "⚠️",
    error: "❌",
    success: "✅",
  };

  return (
    <div className={`my-4 rounded border-l-4 p-4 ${styles[type]}`}>
      {title && (
        <p className="mb-2 font-semibold">
          {icons[type]} {title}
        </p>
      )}
      <div>{children}</div>
    </div>
  );
}

/**
 * 解析自定义指令
 *
 * 支持语法：
 * :::chart{config}
 * :::table{data}
 * :::callout{type="info", title="标题"}
 */

export function parseCustomDirectives(content: string): string {
  // 这个函数可以预处理 Markdown 内容
  // 将自定义指令转换为 HTML 或特殊标记

  // 示例：将 :::chart{...} 转换为特殊的代码块
  let processed = content;

  // 处理图表指令
  processed = processed.replace(
    /:::chart\s*\{([\s\S]*?)\}\s*:::/g,
    (match, config) => {
      return `\`\`\`chart\n${config}\n\`\`\``;
    },
  );

  // 处理表格指令
  processed = processed.replace(
    /:::table\s*\{([\s\S]*?)\}\s*:::/g,
    (match, data) => {
      return `\`\`\`table\n${data}\n\`\`\``;
    },
  );

  return processed;
}

/**
 * 扩展的 Markdown 组件映射
 *
 * 可以在 MarkdownText 组件中使用
 */

export const extendedMarkdownComponents: Record<string, ComponentType<any>> = {
  // 可以添加更多自定义组件
  chart: ChartDirective,
  table: TableDirective,
  callout: Callout,
};

/**
 * 使用示例：
 *
 * // 在 markdown-text.tsx 中
 * import { extendedMarkdownComponents } from './markdown-extensions';
 *
 * <ReactMarkdown
 *   components={{
 *     ...defaultComponents,
 *     ...extendedMarkdownComponents
 *   }}
 * >
 *   {children}
 * </ReactMarkdown>
 */
