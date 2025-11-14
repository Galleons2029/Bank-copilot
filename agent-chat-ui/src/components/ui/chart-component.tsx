"use client";

import { useEffect, useRef } from "react";

/**
 * 真实的图表组件
 * 使用 ECharts 渲染图表
 */
export function ChartComponent({
  config,
  className,
}: {
  config: any;
  className?: string;
}) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<any>(null);

  useEffect(() => {
    if (!chartRef.current) return;

    // 动态导入 ECharts
    import("echarts").then((echarts) => {
      if (!chartRef.current) return;

      // 初始化图表
      if (!chartInstance.current) {
        chartInstance.current = echarts.init(chartRef.current);
      }

      // 设置配置
      if (config.option) {
        chartInstance.current.setOption(config.option);
      }

      // 响应式
      const handleResize = () => {
        chartInstance.current?.resize();
      };
      window.addEventListener("resize", handleResize);

      return () => {
        window.removeEventListener("resize", handleResize);
      };
    });

    // 清理
    return () => {
      chartInstance.current?.dispose();
      chartInstance.current = null;
    };
  }, [config]);

  return (
    <div className={className}>
      <div
        ref={chartRef}
        className="h-[400px] w-full"
        style={{ minHeight: "400px" }}
      />
    </div>
  );
}
