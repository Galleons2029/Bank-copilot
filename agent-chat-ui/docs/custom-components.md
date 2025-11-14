# 🎨 前端自定义组件集成指南

本项目支持两种方式在消息中渲染自定义组件：

1. **后端传递 UI 组件**（原有方式）- 通过 UIMessage 发送 shadow root
2. **前端组件注册表**（新增）- 基于消息内容/元数据自动触发前端组件

## 📦 方式1：前端组件注册表（推荐）

### 优势

- ✅ 无需后端传递复杂的 shadow root HTML
- ✅ 前端完全控制组件样式和交互
- ✅ 更好的类型安全和开发体验
- ✅ 支持代码分割和懒加载

### 使用方式

#### 方法 A：通过 `additional_kwargs` 传递组件配置

**Python 后端示例（标准格式）：**

```python
from langchain_core.messages import AIMessage

# 创建带组件配置的消息（标准格式）
message = AIMessage(
    content="这是一个图表可视化结果：",
    additional_kwargs={
        "component": {
            "type": "chart",
            "data": {
                "option": {
                    "title": {"text": "销售数据"},
                    "xAxis": {
                        "type": "category",
                        "data": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                    },
                    "yAxis": {"type": "value"},
                    "series": [{
                        "data": [120, 200, 150, 80, 70, 110, 130],
                        "type": "bar"
                    }]
                }
            }
        }
    }
)

return {"messages": [message]}
```

**Python 后端示例（简化格式，直接传递 type 和 data）：**

```python
from langchain_core.messages import AIMessage

# 创建带组件配置的消息（简化格式，适用于 echart_agent）
message = AIMessage(
    content="这是一个图表可视化结果",
    additional_kwargs={
        "type": "image",
        "data": {
            "layout": "carousel",
            "images": [
                {
                    "url": "data:image/png;base64,iVBORw0KGgo...",
                    "alt": "图表 1"
                },
                {
                    "url": "data:image/png;base64,iVBORw0KGgo...",
                    "alt": "图表 2"
                }
            ],
            "caption": "图表可视化结果"
        }
    }
)

return {"messages": [message]}
```

#### 方法 B：通过 Tool Call 触发组件

**Python 后端示例：**

```python
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage

@tool
def render_chart(chart_config: dict) -> str:
    """渲染图表组件

    Args:
        chart_config: ECharts 配置对象
    """
    return "Chart rendered successfully"

# 在 agent 中使用
message = AIMessage(
    content="",
    tool_calls=[{
        "name": "render_chart",
        "args": {
            "option": {
                "title": {"text": "用户增长"},
                "xAxis": {"type": "category", "data": ["Jan", "Feb", "Mar"]},
                "yAxis": {"type": "value"},
                "series": [{"data": [100, 150, 200], "type": "line"}]
            }
        },
        "id": "call_123",
        "type": "tool_call"
    }]
)
```

#### 方法 C：通过 Markdown 代码块

**Python 后端示例：**

````python
from langchain_core.messages import AIMessage

# 在消息内容中嵌入特殊 Markdown 代码块
message = AIMessage(
    content="""
分析结果如下：

\```chart
{
  "option": {
    "title": {"text": "月度销售额"},
    "xAxis": {"type": "category", "data": ["1月", "2月", "3月", "4月"]},
    "yAxis": {"type": "value"},
    "series": [{
      "name": "销售额",
      "data": [8200, 9320, 9010, 13420],
      "type": "line"
    }]
  }
}
\```

以上是数据可视化结果。
"""
)
````

### 支持的组件类型

#### 1. 图表组件 (`type: "chart"`)

```python
{
    "type": "chart",
    "data": {
        "option": {
            # ECharts 配置对象
            "title": {"text": "标题"},
            "xAxis": {"type": "category", "data": ["A", "B", "C"]},
            "yAxis": {"type": "value"},
            "series": [{"data": [10, 20, 30], "type": "bar"}]
        }
    }
}
```

#### 2. 表格组件 (`type: "table"`)

```python
{
    "type": "table",
    "data": {
        "headers": ["姓名", "年龄", "城市"],
        "rows": [
            ["张三", 25, "北京"],
            ["李四", 30, "上海"],
            ["王五", 28, "深圳"]
        ]
    }
}
```

#### 3. 图片组件 (`type: "image"`)

**单张图片：**

```python
{
    "type": "image",
    "data": {
        "url": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&h=500&fit=crop",
        "alt": "数据分析图表",
        "caption": "商业数据分析仪表板"
    }
}
```

**多张图片（网格布局）：**

```python
{
    "type": "image",
    "data": {
        "layout": "grid",  # 可选：grid（默认）或 carousel
        "images": [
            {
                "url": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=600&h=400&fit=crop",
                "alt": "销售趋势图",
                "caption": "📈 2024年销售趋势"
            },
            {
                "url": "https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=600&h=400&fit=crop",
                "alt": "用户增长图",
                "caption": "👥 用户增长分析"
            },
            {
                "url": "https://images.unsplash.com/photo-1504868584819-f8e8b4b6d7e3?w=600&h=400&fit=crop",
                "alt": "收入分布图",
                "caption": "💰 收入分布情况"
            }
        ],
        "caption": "综合数据分析报告"
    }
}
```

**多张图片（轮播图）：**

```python
{
    "type": "image",
    "data": {
        "layout": "carousel",
        "images": [
            {
                "url": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&h=500&fit=crop",
                "alt": "销售数据分析",
                "caption": "2024年销售趋势分析"
            },
            {
                "url": "https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=800&h=500&fit=crop",
                "alt": "业务增长图表",
                "caption": "业务增长指标概览"
            }
        ],
        "caption": "数据可视化报告"
    }
}
```

**布局说明：**

- `grid`（默认）：网格布局，2张图片显示为2列，3张显示为3列，4张及以上显示为2列网格
  - 适合同时展示多张图表，便于对比查看
  - 每张图片可以有独立的 caption
- `carousel`：轮播图布局，支持横向滑动查看
  - 最大宽度限制为 768px（max-w-3xl），避免图片过大
  - 图片高度限制为 500px（max-h-[500px]），保持合理比例
  - 支持触摸滑动和鼠标拖动
  - 适合展示多张图表的详细内容

#### 4. 文件组件 (`type: "file"`)

```python
{
    "type": "file",
    "data": {
        "name": "report.pdf",
        "size": 1024000,  # 字节
        "url": "https://example.com/download/report.pdf"
    }
}
```

## 🔧 方式2：后端传递 UI 组件（原有方式）

如果你需要更复杂的动态组件，仍然可以使用原有的 UIMessage 方式：

```python
from langgraph.types import Command

# 发送自定义 UI 事件
return Command(
    update={
        "messages": [AIMessage(content="处理完成", id="msg_123")],
    },
    custom_events=[
        {
            "type": "ui",
            "id": "ui_component_123",
            "content": "<custom-chart-preview>...</custom-chart-preview>",
            "metadata": {
                "message_id": "msg_123"
            }
        }
    ]
)
```

## 🎯 完整示例

### 示例1：EChart 图表

```python
# src/react_agent/echart_agent.py

from typing import Annotated
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

class State(TypedDict):
    messages: Annotated[list, "消息列表"]

@tool
def create_chart(data: dict, chart_type: str = "bar") -> str:
    """创建数据图表

    Args:
        data: 数据对象，包含 categories 和 values
        chart_type: 图表类型 (bar, line, pie)
    """
    return f"Chart created: {chart_type}"

def chart_agent(state: State):
    """图表生成代理"""
    messages = state["messages"]
    last_message = messages[-1]

    # 模拟数据分析
    chart_data = {
        "option": {
            "title": {"text": "数据分析结果"},
            "xAxis": {
                "type": "category",
                "data": ["产品A", "产品B", "产品C", "产品D"]
            },
            "yAxis": {"type": "value"},
            "series": [{
                "name": "销量",
                "data": [320, 240, 360, 180],
                "type": "bar"
            }]
        }
    }

    # 使用 additional_kwargs
    response = AIMessage(
        content="我已经分析了数据，以下是可视化结果：",
        additional_kwargs={
            "component": {
                "type": "chart",
                "data": chart_data
            }
        }
    )

    return {"messages": [response]}

# 构建图
graph = StateGraph(State)
graph.add_node("agent", chart_agent)
graph.set_entry_point("agent")
graph.set_finish_point("agent")

app = graph.compile()
```

### 示例2：多图片展示

```python
def image_gallery_agent(state: State):
    """图片画廊代理 - 展示多张图片"""

    # 网格布局示例
    response = AIMessage(
        content="这是项目的设计稿展示：",
        additional_kwargs={
            "component": {
                "type": "image",
                "data": {
                    "layout": "grid",
                    "images": [
                        {
                            "url": "https://example.com/design1.jpg",
                            "alt": "首页设计",
                            "caption": "首页 - 桌面版"
                        },
                        {
                            "url": "https://example.com/design2.jpg",
                            "alt": "详情页设计",
                            "caption": "详情页 - 桌面版"
                        },
                        {
                            "url": "https://example.com/design3.jpg",
                            "alt": "移动端设计",
                            "caption": "首页 - 移动版"
                        }
                    ],
                    "caption": "UI 设计方案 v1.0"
                }
            }
        }
    )

    return {"messages": [response]}

def image_carousel_agent(state: State):
    """图片轮播代理 - 展示轮播图"""

    response = AIMessage(
        content="这是产品的不同角度展示：",
        additional_kwargs={
            "component": {
                "type": "image",
                "data": {
                    "layout": "carousel",
                    "images": [
                        {"url": "https://example.com/product-front.jpg", "alt": "正面"},
                        {"url": "https://example.com/product-side.jpg", "alt": "侧面"},
                        {"url": "https://example.com/product-back.jpg", "alt": "背面"},
                        {"url": "https://example.com/product-detail.jpg", "alt": "细节"}
                    ],
                    "caption": "产品360度展示"
                }
            }
        }
    )

    return {"messages": [response]}
```

## 🎨 扩展自定义组件

如果你需要添加新的组件类型：

### 1. 在注册表中添加新组件

编辑 `src/components/thread/messages/custom-component-registry.tsx`:

```typescript
// 添加新的组件类型
export type CustomComponentType =
  | "chart"
  | "table"
  | "image"
  | "file"
  | "custom"
  | "video"  // 新增
  | "map";   // 新增

// 在 CustomComponentRenderer 中添加渲染逻辑
export function CustomComponentRenderer({ config }: { config: ComponentConfig }) {
  switch (config.type) {
    // ... existing cases

    case "video":
      return (
        <div className="my-4">
          <video controls className="w-full rounded-lg">
            <source src={config.data.url} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </div>
      );

    case "map":
      return (
        <div className="my-4 h-96 rounded-lg overflow-hidden">
          {/* 集成地图组件，如 react-leaflet */}
        </div>
      );

    default:
      // ...
  }
}
```

### 2. 后端发送新组件

```python
message = AIMessage(
    content="视频分析结果：",
    additional_kwargs={
        "component": {
            "type": "video",
            "data": {
                "url": "https://example.com/video.mp4",
                "thumbnail": "https://example.com/thumb.jpg"
            }
        }
    }
)
```

## 📊 最佳实践

1. **优先使用前端组件注册表**

   - 更好的性能和用户体验
   - 更容易维护和测试

2. **为复杂交互使用后端 UI 组件**

   - 需要实时数据更新
   - 复杂的状态管理

3. **合理使用懒加载**

   - 大型图表库使用 dynamic import
   - 提升初始加载速度

4. **提供加载状态**

   - 使用 Suspense 边界
   - 友好的加载提示

5. **错误处理**
   - 验证数据格式
   - 提供友好的错误提示

## 🔍 调试技巧

### 查看消息结构

在浏览器控制台中：

```javascript
// 查看所有消息
console.log(stream.messages);

// 查看特定消息的 additional_kwargs
console.log(stream.messages[0].additional_kwargs);

// 查看 UI 组件列表
console.log(stream.values.ui);
```

### 测试组件渲染

在 `custom-component-registry.tsx` 中添加调试日志：

```typescript
export function extractComponentConfig(message: Message): ComponentConfig | null {
  console.log('Extracting component from message:', message);

  const config = /* ... */;

  if (config) {
    console.log('Found component config:', config);
  }

  return config;
}
```

## 🚀 性能优化

1. **使用 React.memo** 避免不必要的重渲染
2. **懒加载重型组件** 如 ECharts
3. **虚拟化长列表** 使用 react-window
4. **缓存数据** 避免重复计算

## ❓ FAQ

**Q: 前端组件和后端 UI 组件可以同时使用吗？**
A: 可以！两种方式可以共存，会同时渲染。

**Q: 如何传递大量数据？**
A: 对于大数据集，建议使用 URL 引用，而不是直接在消息中传递。

**Q: 支持自定义样式吗？**
A: 支持！你可以在组件配置中传递 className 或 style。

**Q: 如何处理组件交互事件？**
A: 可以在组件中使用 `useStreamContext()` 访问 stream 对象，调用后端 API。
