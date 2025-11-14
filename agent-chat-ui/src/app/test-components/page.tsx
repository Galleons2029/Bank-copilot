"use client";

import { useState } from "react";
import {
  extractComponentConfig,
  CustomComponentRenderer,
  type ComponentConfig,
} from "@/components/thread/messages/custom-component-registry";
import { Message } from "@langchain/langgraph-sdk";
import { MarkdownText } from "@/components/thread/markdown-text";
import { MermaidDiagram } from "@/components/thread/mermaid-diagram";

/**
 * 组件测试页面
 * 用于测试自定义组件渲染功能
 */
export default function TestComponentsPage() {
  const [activeTab, setActiveTab] = useState<
    | "chart"
    | "table"
    | "image"
    | "image-carousel"
    | "image-grid"
    | "file"
    | "markdown"
    | "mermaid"
  >("chart");

  // 测试数据：图表组件
  const chartConfig: ComponentConfig = {
    type: "chart",
    data: {
      option: {
        title: { text: "月度销售数据", left: "center" },
        tooltip: { trigger: "axis" },
        xAxis: {
          type: "category",
          data: ["1月", "2月", "3月", "4月", "5月", "6月"],
        },
        yAxis: { type: "value", name: "销售额（万元）" },
        series: [
          {
            name: "销售额",
            data: [120, 200, 150, 180, 220, 280],
            type: "line",
            smooth: true,
            itemStyle: { color: "#5470c6" },
          },
        ],
      },
    },
  };

  // 测试数据：表格组件
  const tableConfig: ComponentConfig = {
    type: "table",
    data: {
      headers: ["产品名称", "销量", "增长率", "库存状态"],
      rows: [
        ["iPhone 15", "12,500", "+15%", "充足"],
        ["MacBook Pro", "8,900", "+22%", "偏低"],
        ["iPad Air", "15,200", "+8%", "充足"],
        ["AirPods Pro", "25,600", "+35%", "紧张"],
      ],
    },
  };

  // 测试数据：图片组件（单张）
  const imageConfig: ComponentConfig = {
    type: "image",
    data: {
      url: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&h=500&fit=crop",
      alt: "数据分析图表",
      caption: "商业数据分析仪表板",
    },
  };

  // 测试数据：图片组件（多张 - carousel 布局，模拟 echart_agent）
  const imageCarouselConfig: ComponentConfig = {
    type: "image",
    data: {
      layout: "carousel",
      images: [
        {
          url: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&h=500&fit=crop",
          alt: "销售数据分析",
          caption: "2024年销售趋势分析",
        },
        {
          url: "https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=800&h=500&fit=crop",
          alt: "业务增长图表",
          caption: "业务增长指标概览",
        },
        {
          url: "https://images.unsplash.com/photo-1504868584819-f8e8b4b6d7e3?w=800&h=500&fit=crop",
          alt: "财务报表",
          caption: "季度财务报表分析",
        },
      ],
      caption: "数据可视化报告（轮播查看）",
    },
  };

  // 测试数据：图片组件（多张 - grid 布局）
  const imageGridConfig: ComponentConfig = {
    type: "image",
    data: {
      layout: "grid",
      images: [
        {
          url: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=600&h=400&fit=crop",
          alt: "销售趋势图",
          caption: "📈 2024年销售趋势",
        },
        {
          url: "https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=600&h=400&fit=crop",
          alt: "用户增长图",
          caption: "👥 用户增长分析",
        },
        {
          url: "https://images.unsplash.com/photo-1504868584819-f8e8b4b6d7e3?w=600&h=400&fit=crop",
          alt: "收入分布图",
          caption: "💰 收入分布情况",
        },
        {
          url: "https://images.unsplash.com/photo-1543286386-713bdd548da4?w=600&h=400&fit=crop",
          alt: "地区对比图",
          caption: "🌍 各地区业绩对比",
        },
      ],
      caption: "综合数据分析报告（网格对比）",
    },
  };

  // 测试数据：文件组件
  const fileConfig: ComponentConfig = {
    type: "file",
    data: {
      name: "sales_report_2024.pdf",
      size: 2048576, // 2MB
      url: "#download",
    },
  };

  // 测试消息：Markdown 代码块提取
  const markdownMessage: Partial<Message> = {
    content: `
分析完成，以下是数据可视化结果：

\`\`\`chart
{
  "option": {
    "title": {"text": "用户增长趋势"},
    "xAxis": {
      "type": "category",
      "data": ["Jan", "Feb", "Mar", "Apr", "May"]
    },
    "yAxis": {"type": "value"},
    "series": [{
      "name": "用户数",
      "data": [1200, 1500, 1800, 2100, 2500],
      "type": "bar",
      "itemStyle": {"color": "#91cc75"}
    }]
  }
}
\`\`\`

从图表可以看出，用户数呈现持续增长态势。
`,
  };

  const markdownComponentConfig = extractComponentConfig(
    markdownMessage as Message,
  );

  // 测试消息：additional_kwargs（标准格式）
  const additionalKwargsMessage: Partial<Message> = {
    content: "数据分析结果如下：",
    additional_kwargs: {
      component: chartConfig,
    },
  };

  const additionalKwargsConfig = extractComponentConfig(
    additionalKwargsMessage as Message,
  );

  // 测试消息：additional_kwargs（echart_agent 格式）
  const echartAgentMessage: Partial<Message> = {
    content: "这是一个图表可视化结果",
    additional_kwargs: {
      type: "image",
      data: {
        layout: "carousel",
        images: [
          {
            url: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&h=500&fit=crop",
            alt: "ECharts 图表 1",
            caption: "销售数据可视化",
          },
          {
            url: "https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=800&h=500&fit=crop",
            alt: "ECharts 图表 2",
            caption: "业务指标分析",
          },
        ],
        caption: "ECharts 图表可视化结果",
      },
    },
  };

  const echartAgentConfig = extractComponentConfig(
    echartAgentMessage as Message,
  );

  // Mermaid 图表示例
  const mermaidFlowchart = `graph TD
    A[开始] --> B{是否需要检索?}
    B -->|是| C[执行RAG检索]
    B -->|否| D[直接回答]
    C --> E[生成答案]
    D --> E
    E --> F[结束]`;

  const mermaidSequence = `sequenceDiagram
    participant U as 用户
    participant A as AI助手
    participant R as RAG系统
    participant D as 数据库
    
    U->>A: 提出问题
    A->>R: 触发检索
    R->>D: 查询相关文档
    D-->>R: 返回文档
    R-->>A: 返回上下文
    A->>A: 生成答案
    A-->>U: 返回答案`;

  const mermaidGantt = `gantt
    title 项目开发进度
    dateFormat YYYY-MM-DD
    section 设计阶段
    需求分析      :2024-01-01, 7d
    UI设计        :2024-01-08, 5d
    section 开发阶段
    前端开发      :2024-01-13, 14d
    后端开发      :2024-01-13, 14d
    section 测试阶段
    集成测试      :2024-01-27, 5d
    上线部署      :2024-02-01, 2d`;

  const mermaidPie = `pie title 技术栈占比
    "React" : 35
    "TypeScript" : 25
    "Python" : 20
    "Docker" : 15
    "其他" : 5`;

  const mermaidClassDiagram = `classDiagram
    class Agent {
        +String name
        +List tools
        +execute()
        +think()
    }
    class RAGAgent {
        +VectorStore vectorStore
        +retrieve()
        +answer()
    }
    class Tool {
        +String name
        +run()
    }
    Agent <|-- RAGAgent
    Agent "1" --> "*" Tool : uses`;

  // Markdown 中的 Mermaid 测试
  const markdownWithMermaid = `## 系统架构图

下面是我们系统的核心流程：

\`\`\`mermaid
graph LR
    A[用户请求] --> B[API网关]
    B --> C[LangGraph代理]
    C --> D{需要工具?}
    D -->|是| E[调用工具]
    D -->|否| F[LLM生成]
    E --> F
    F --> G[返回结果]
\`\`\`

这个架构支持以下功能：

1. **智能路由**：根据用户请求智能选择工具
2. **RAG检索**：从向量数据库检索相关内容
3. **多轮对话**：支持上下文记忆

### 时序图示例

\`\`\`mermaid
sequenceDiagram
    用户->>系统: 发送问题
    系统->>LLM: 分析问题
    LLM->>系统: 返回意图
    系统->>向量DB: 检索相关文档
    向量DB->>系统: 返回文档
    系统->>LLM: 生成答案
    LLM->>系统: 返回答案
    系统->>用户: 显示答案
\`\`\`

这就是完整的工作流程！`;

  const tabs = [
    { id: "chart", label: "📊 图表组件", config: chartConfig },
    { id: "table", label: "📋 表格组件", config: tableConfig },
    { id: "image", label: "🖼️ 单张图片", config: imageConfig },
    {
      id: "image-carousel",
      label: "🎠 轮播布局",
      config: imageCarouselConfig,
    },
    {
      id: "image-grid",
      label: "🔲 网格布局",
      config: imageGridConfig,
    },
    { id: "file", label: "📁 文件组件", config: fileConfig },
    {
      id: "markdown",
      label: "📝 Markdown 提取",
      config: markdownComponentConfig,
    },
    { id: "mermaid", label: "🔷 Mermaid 图表", config: null },
  ];

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="mx-auto max-w-6xl">
        {/* 页面标题 */}
        <div className="mb-8">
          <h1 className="mb-2 text-3xl font-bold text-gray-900">
            🧪 前端组件测试页面
          </h1>
          <p className="text-gray-600">测试自定义组件的渲染和参数配置</p>
        </div>

        {/* 标签页导航 */}
        <div className="mb-6 flex gap-2 overflow-x-auto border-b">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`rounded-t-lg px-4 py-2 font-medium whitespace-nowrap transition-colors ${
                activeTab === tab.id
                  ? "border-b-2 border-blue-600 bg-white text-blue-600"
                  : "text-gray-600 hover:bg-gray-100"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* 内容区域 */}
        <div className="rounded-lg bg-white p-6 shadow-md">
          {/* 图表组件测试 */}
          {activeTab === "chart" && (
            <div>
              <h2 className="mb-4 text-xl font-semibold">图表组件测试</h2>
              <div className="mb-6 rounded border border-gray-200 bg-gray-50 p-4">
                <h3 className="mb-2 text-sm font-medium text-gray-700">
                  配置参数：
                </h3>
                <pre className="overflow-auto text-xs">
                  {JSON.stringify(chartConfig, null, 2)}
                </pre>
              </div>
              <div className="mb-4">
                <h3 className="mb-2 text-sm font-medium text-gray-700">
                  渲染结果：
                </h3>
                <CustomComponentRenderer config={chartConfig} />
              </div>
            </div>
          )}

          {/* 表格组件测试 */}
          {activeTab === "table" && (
            <div>
              <h2 className="mb-4 text-xl font-semibold">表格组件测试</h2>
              <div className="mb-6 rounded border border-gray-200 bg-gray-50 p-4">
                <h3 className="mb-2 text-sm font-medium text-gray-700">
                  配置参数：
                </h3>
                <pre className="overflow-auto text-xs">
                  {JSON.stringify(tableConfig, null, 2)}
                </pre>
              </div>
              <div className="mb-4">
                <h3 className="mb-2 text-sm font-medium text-gray-700">
                  渲染结果：
                </h3>
                <CustomComponentRenderer config={tableConfig} />
              </div>
            </div>
          )}

          {/* 图片组件测试 - 单张 */}
          {activeTab === "image" && (
            <div>
              <h2 className="mb-4 text-xl font-semibold">
                图片组件测试（单张）
              </h2>
              <div className="mb-6 rounded border border-gray-200 bg-gray-50 p-4">
                <h3 className="mb-2 text-sm font-medium text-gray-700">
                  配置参数：
                </h3>
                <pre className="overflow-auto text-xs">
                  {JSON.stringify(imageConfig, null, 2)}
                </pre>
              </div>
              <div className="mb-4">
                <h3 className="mb-2 text-sm font-medium text-gray-700">
                  渲染结果：
                </h3>
                <CustomComponentRenderer config={imageConfig} />
              </div>
            </div>
          )}

          {/* 图片组件测试 - 多张轮播 */}
          {activeTab === "image-carousel" && (
            <div>
              <h2 className="mb-4 text-xl font-semibold">
                图片组件测试（轮播布局）
              </h2>
              <div className="mb-6 rounded border border-gray-200 bg-gray-50 p-4">
                <h3 className="mb-2 text-sm font-medium text-gray-700">
                  配置参数：
                </h3>
                <pre className="overflow-auto text-xs">
                  {JSON.stringify(imageCarouselConfig, null, 2)}
                </pre>
              </div>
              <div className="mb-4">
                <h3 className="mb-2 text-sm font-medium text-gray-700">
                  渲染结果：
                </h3>
                <CustomComponentRenderer config={imageCarouselConfig} />
              </div>
              <div className="mt-4 rounded border border-blue-200 bg-blue-50 p-4">
                <h3 className="mb-2 text-sm font-medium text-blue-700">
                  💡 说明
                </h3>
                <ul className="list-inside list-disc space-y-1 text-sm text-blue-800">
                  <li>轮播布局适合展示多张图表，用户可以左右滑动查看</li>
                  <li>最大宽度限制为 768px（max-w-3xl），避免图片过大</li>
                  <li>图片高度限制为 500px（max-h-[500px]），保持合理比例</li>
                  <li>支持触摸滑动和鼠标拖动</li>
                </ul>
              </div>
            </div>
          )}

          {/* 图片组件测试 - 多张网格 */}
          {activeTab === "image-grid" && (
            <div>
              <h2 className="mb-4 text-xl font-semibold">
                图片组件测试（网格布局）
              </h2>
              <div className="mb-6 rounded border border-gray-200 bg-gray-50 p-4">
                <h3 className="mb-2 text-sm font-medium text-gray-700">
                  配置参数：
                </h3>
                <pre className="overflow-auto text-xs">
                  {JSON.stringify(imageGridConfig, null, 2)}
                </pre>
              </div>
              <div className="mb-4">
                <h3 className="mb-2 text-sm font-medium text-gray-700">
                  渲染结果：
                </h3>
                <CustomComponentRenderer config={imageGridConfig} />
              </div>
              <div className="mt-4 rounded border border-green-200 bg-green-50 p-4">
                <h3 className="mb-2 text-sm font-medium text-green-700">
                  💡 说明
                </h3>
                <ul className="list-inside list-disc space-y-1 text-sm text-green-800">
                  <li>网格布局适合同时展示多张图表，便于对比</li>
                  <li>2张图片：2列布局</li>
                  <li>3张图片：3列布局</li>
                  <li>4张及以上：2列布局（每行2张）</li>
                  <li>每张图片可以有独立的标题（caption）</li>
                </ul>
              </div>
            </div>
          )}

          {/* 文件组件测试 */}
          {activeTab === "file" && (
            <div>
              <h2 className="mb-4 text-xl font-semibold">文件组件测试</h2>
              <div className="mb-6 rounded border border-gray-200 bg-gray-50 p-4">
                <h3 className="mb-2 text-sm font-medium text-gray-700">
                  配置参数：
                </h3>
                <pre className="overflow-auto text-xs">
                  {JSON.stringify(fileConfig, null, 2)}
                </pre>
              </div>
              <div className="mb-4">
                <h3 className="mb-2 text-sm font-medium text-gray-700">
                  渲染结果：
                </h3>
                <CustomComponentRenderer config={fileConfig} />
              </div>
            </div>
          )}

          {/* Markdown 提取测试 */}
          {activeTab === "markdown" && (
            <div>
              <h2 className="mb-4 text-xl font-semibold">
                Markdown 代码块提取测试
              </h2>
              <div className="mb-6 rounded border border-gray-200 bg-gray-50 p-4">
                <h3 className="mb-2 text-sm font-medium text-gray-700">
                  输入消息：
                </h3>
                <pre className="overflow-auto text-xs">
                  {typeof markdownMessage.content === "string"
                    ? markdownMessage.content
                    : JSON.stringify(markdownMessage.content, null, 2)}
                </pre>
              </div>
              <div className="mb-6 rounded border border-blue-200 bg-blue-50 p-4">
                <h3 className="mb-2 text-sm font-medium text-blue-700">
                  提取结果：
                </h3>
                <pre className="overflow-auto text-xs">
                  {markdownComponentConfig
                    ? JSON.stringify(markdownComponentConfig, null, 2)
                    : "未提取到组件配置"}
                </pre>
              </div>
              {markdownComponentConfig && (
                <div className="mb-4">
                  <h3 className="mb-2 text-sm font-medium text-gray-700">
                    渲染结果：
                  </h3>
                  <CustomComponentRenderer config={markdownComponentConfig} />
                </div>
              )}
            </div>
          )}

          {/* Mermaid 图表测试 */}
          {activeTab === "mermaid" && (
            <div>
              <h2 className="mb-4 text-xl font-semibold">
                Mermaid 图表渲染测试
              </h2>

              {/* 流程图 */}
              <div className="mb-8">
                <h3 className="mb-3 text-lg font-medium">
                  1. 流程图 (Flowchart)
                </h3>
                <div className="rounded-lg border border-gray-200 bg-gray-50 p-4">
                  <MermaidDiagram chart={mermaidFlowchart} />
                </div>
              </div>

              {/* 序列图 */}
              <div className="mb-8">
                <h3 className="mb-3 text-lg font-medium">
                  2. 序列图 (Sequence Diagram)
                </h3>
                <div className="rounded-lg border border-gray-200 bg-gray-50 p-4">
                  <MermaidDiagram chart={mermaidSequence} />
                </div>
              </div>

              {/* 甘特图 */}
              <div className="mb-8">
                <h3 className="mb-3 text-lg font-medium">
                  3. 甘特图 (Gantt Chart)
                </h3>
                <div className="rounded-lg border border-gray-200 bg-gray-50 p-4">
                  <MermaidDiagram chart={mermaidGantt} />
                </div>
              </div>

              {/* 饼图 */}
              <div className="mb-8">
                <h3 className="mb-3 text-lg font-medium">
                  4. 饼图 (Pie Chart)
                </h3>
                <div className="rounded-lg border border-gray-200 bg-gray-50 p-4">
                  <MermaidDiagram chart={mermaidPie} />
                </div>
              </div>

              {/* 类图 */}
              <div className="mb-8">
                <h3 className="mb-3 text-lg font-medium">
                  5. 类图 (Class Diagram)
                </h3>
                <div className="rounded-lg border border-gray-200 bg-gray-50 p-4">
                  <MermaidDiagram chart={mermaidClassDiagram} />
                </div>
              </div>

              {/* Markdown 中的 Mermaid */}
              <div className="mb-8">
                <h3 className="mb-3 text-lg font-medium">
                  6. Markdown 中的 Mermaid（集成测试）
                </h3>
                <div className="mb-4 rounded border border-purple-200 bg-purple-50 p-4">
                  <h4 className="mb-2 text-sm font-medium text-purple-700">
                    💡 说明
                  </h4>
                  <p className="text-sm text-purple-800">
                    这个测试展示了在 Markdown 内容中使用 ```mermaid
                    代码块的效果，模拟 AI
                    助手在回答中包含流程图、序列图等可视化内容的场景。每个图表都自带"查看源代码"按钮。
                  </p>
                </div>
                <div className="rounded-lg border border-gray-200 bg-gray-50 p-4">
                  <MarkdownText>{markdownWithMermaid}</MarkdownText>
                </div>
              </div>

              {/* 支持的图表类型说明 */}
              <div className="rounded-lg border border-blue-200 bg-blue-50 p-6">
                <h3 className="mb-3 text-lg font-semibold text-blue-900">
                  📊 Mermaid 支持的图表类型
                </h3>
                <ul className="space-y-2 text-sm text-blue-800">
                  <li>
                    <strong>流程图 (Flowchart)</strong>：展示流程和决策逻辑
                  </li>
                  <li>
                    <strong>序列图 (Sequence Diagram)</strong>
                    ：展示对象间的交互时序
                  </li>
                  <li>
                    <strong>甘特图 (Gantt Chart)</strong>：展示项目时间规划
                  </li>
                  <li>
                    <strong>饼图 (Pie Chart)</strong>：展示数据占比
                  </li>
                  <li>
                    <strong>类图 (Class Diagram)</strong>：展示面向对象的类结构
                  </li>
                  <li>
                    <strong>状态图 (State Diagram)</strong>：展示状态转换
                  </li>
                  <li>
                    <strong>ER图 (Entity Relationship)</strong>：展示数据库关系
                  </li>
                  <li>
                    <strong>用户旅程图 (User Journey)</strong>：展示用户体验流程
                  </li>
                </ul>
              </div>
            </div>
          )}
        </div>

        {/* additional_kwargs 测试 - 标准格式 */}
        <div className="mt-8 rounded-lg bg-white p-6 shadow-md">
          <h2 className="mb-4 text-xl font-semibold">
            additional_kwargs 提取测试（标准格式）
          </h2>
          <div className="mb-6 rounded border border-gray-200 bg-gray-50 p-4">
            <h3 className="mb-2 text-sm font-medium text-gray-700">
              模拟消息对象：
            </h3>
            <pre className="overflow-auto text-xs">
              {JSON.stringify(additionalKwargsMessage, null, 2)}
            </pre>
          </div>
          <div className="mb-6 rounded border border-green-200 bg-green-50 p-4">
            <h3 className="mb-2 text-sm font-medium text-green-700">
              提取结果：
            </h3>
            <pre className="overflow-auto text-xs">
              {additionalKwargsConfig
                ? JSON.stringify(additionalKwargsConfig, null, 2)
                : "未提取到组件配置"}
            </pre>
          </div>
          {additionalKwargsConfig && (
            <div>
              <h3 className="mb-2 text-sm font-medium text-gray-700">
                渲染结果：
              </h3>
              <CustomComponentRenderer config={additionalKwargsConfig} />
            </div>
          )}
        </div>

        {/* additional_kwargs 测试 - echart_agent 格式 */}
        <div className="mt-8 rounded-lg bg-white p-6 shadow-md">
          <h2 className="mb-4 text-xl font-semibold">
            additional_kwargs 提取测试（echart_agent 格式）
          </h2>
          <div className="mb-6 rounded border border-gray-200 bg-gray-50 p-4">
            <h3 className="mb-2 text-sm font-medium text-gray-700">
              模拟消息对象（后端格式）：
            </h3>
            <pre className="overflow-auto text-xs">
              {JSON.stringify(echartAgentMessage, null, 2)}
            </pre>
          </div>
          <div className="mb-6 rounded border border-purple-200 bg-purple-50 p-4">
            <h3 className="mb-2 text-sm font-medium text-purple-700">
              提取结果：
            </h3>
            <pre className="overflow-auto text-xs">
              {echartAgentConfig
                ? JSON.stringify(echartAgentConfig, null, 2)
                : "未提取到组件配置"}
            </pre>
          </div>
          {echartAgentConfig && (
            <div>
              <h3 className="mb-2 text-sm font-medium text-gray-700">
                渲染结果：
              </h3>
              <CustomComponentRenderer config={echartAgentConfig} />
            </div>
          )}
          <div className="mt-4 rounded border border-blue-200 bg-blue-50 p-4">
            <h3 className="mb-2 text-sm font-medium text-blue-700">💡 说明</h3>
            <p className="text-sm text-blue-800">
              这个测试模拟了后端 echart_agent 返回的消息格式，其中
              additional_kwargs 直接包含 type 和 data 字段，而不是嵌套在
              component 字段中。
            </p>
          </div>
        </div>

        {/* 使用说明 */}
        <div className="mt-8 rounded-lg border border-blue-200 bg-blue-50 p-6">
          <h2 className="mb-3 text-lg font-semibold text-blue-900">
            💡 使用说明
          </h2>
          <ul className="space-y-2 text-sm text-blue-800">
            <li>
              <strong>📊 图表组件</strong>：显示 ECharts 配置的 JSON
              数据（实际使用时会渲染真实图表）
            </li>
            <li>
              <strong>📋 表格组件</strong>：展示表格数据的渲染效果
            </li>
            <li>
              <strong>🖼️ 单张图片</strong>：展示单张图片的加载和显示
            </li>
            <li>
              <strong>🎠 轮播布局</strong>：展示多张图片的轮播效果，最大宽度
              768px，高度限制 500px
            </li>
            <li>
              <strong>🔲 网格布局</strong>：展示多张图片的网格排列，适合对比查看
            </li>
            <li>
              <strong>📁 文件组件</strong>：展示文件信息和下载链接
            </li>
            <li>
              <strong>📝 Markdown 提取</strong>：测试从 Markdown
              代码块中自动提取组件配置
            </li>
            <li>
              <strong>additional_kwargs</strong>：测试从消息的 additional_kwargs
              字段提取配置（支持标准格式和 echart_agent 格式）
            </li>
            <li>
              <strong>🔷 Mermaid 图表</strong>
              ：测试流程图、序列图、甘特图等多种 Mermaid 图表类型，以及在
              Markdown 中的集成使用
            </li>
          </ul>
        </div>

        {/* 返回链接 */}
        <div className="mt-6 text-center">
          <a
            href="/"
            className="text-blue-600 hover:text-blue-800 hover:underline"
          >
            ← 返回主页
          </a>
        </div>
      </div>
    </div>
  );
}
