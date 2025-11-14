# 🎯 前端自定义组件方案总结

## 问题背景

原有架构中，自定义 UI 组件需要后端传递 shadow root HTML，这种方式有以下缺点：

- ❌ 后端需要生成复杂的 HTML 字符串
- ❌ 前端样式和交互难以控制
- ❌ 组件更新需要同时修改前后端代码
- ❌ 类型安全性差，容易出错

## 解决方案对比

### 方案 1：基于消息元数据的组件注册表 ⭐ **推荐**

**位置**：`agent-chat-ui/src/components/thread/messages/custom-component-registry.tsx`

**原理**：

- 前端维护组件注册表
- 后端通过消息的 `additional_kwargs`、`tool_calls` 或特殊 Markdown 语法触发
- 前端自动识别并渲染对应组件

**优点**：

- ✅ 前端完全控制组件实现
- ✅ 类型安全，易于维护
- ✅ 支持代码分割和懒加载
- ✅ 无需后端传递 HTML
- ✅ 支持热更新

**缺点**：

- ⚠️ 需要前后端约定数据格式
- ⚠️ 新增组件需要修改前端代码

**适用场景**：

- 标准化的组件类型（图表、表格、卡片等）
- 需要复杂交互的组件
- 需要良好性能的场景

### 方案 2：Markdown 扩展

**位置**：`agent-chat-ui/src/components/thread/markdown-extensions.tsx`

**原理**：

- 扩展 Markdown 渲染器，支持自定义语法
- 后端在文本中嵌入特殊标记
- 前端解析并渲染自定义组件

**优点**：

- ✅ 对后端侵入最小
- ✅ 可以在纯文本中嵌入组件
- ✅ 易于理解和调试

**缺点**：

- ⚠️ 解析性能开销
- ⚠️ 语法相对受限

**适用场景**：

- 内容为主，组件为辅的场景
- 需要在文本中间插入组件
- 简单的数据展示

### 方案 3：后端传递 Shadow Root（原有方式）

**位置**：使用 `LoadExternalComponent` 组件

**原理**：

- 后端通过 `UIMessage` 发送完整的 HTML/CSS/JS
- 前端使用 Shadow DOM 隔离渲染

**优点**：

- ✅ 最大灵活性
- ✅ 可以动态生成任意组件
- ✅ 不需要修改前端代码

**缺点**：

- ❌ 后端需要生成复杂的 HTML
- ❌ 难以维护和调试
- ❌ 性能开销大
- ❌ 样式隔离可能带来问题

**适用场景**：

- 完全动态的组件
- 第三方系统集成
- 复杂的实时交互

## 📊 使用方式对比

### 后端代码对比

#### 方案 1：组件注册表

```python
# ✅ 简洁清晰
from langchain_core.messages import AIMessage

message = AIMessage(
    content="数据分析结果：",
    additional_kwargs={
        "component": {
            "type": "chart",
            "data": {
                "option": {
                    "xAxis": {"data": ["Mon", "Tue", "Wed"]},
                    "series": [{"data": [120, 200, 150], "type": "bar"}]
                }
            }
        }
    }
)
```

#### 方案 2：Markdown 扩展

````python
# ✅ 最简单
message = AIMessage(
    content="""
分析完成，结果如下：

```chart
{"option": {"xAxis": {"data": ["Mon", "Tue"]}, "series": [{"data": [120, 200]}]}}
````

"""
)

````

#### 方案3：Shadow Root

```python
# ❌ 复杂难维护
from langgraph.types import Command

return Command(
    custom_events=[{
        "type": "ui",
        "content": """
            <script src="echarts.js"></script>
            <div id="chart"></div>
            <script>
                const chart = echarts.init(document.getElementById('chart'));
                chart.setOption({...});
            </script>
        """
    }]
)
````

## 🎨 实际项目集成

### 1. 修改现有 EChart Agent

**文件**：`src/react_agent/sub_agent/echart_agent.py`

```python
# 原代码（使用 shadow root）
from langgraph.types import Command

def echart_node(state):
    # ... 生成图表配置
    return Command(
        custom_events=[{"type": "ui", "content": html_string}]
    )

# ✅ 新代码（使用组件注册表）
from langchain_core.messages import AIMessage

def echart_node(state):
    chart_config = generate_chart_config()  # 生成 ECharts 配置

    return {
        "messages": [
            AIMessage(
                content="数据可视化已生成：",
                additional_kwargs={
                    "component": {
                        "type": "chart",
                        "data": {"option": chart_config}
                    }
                }
            )
        ]
    }
```

### 2. 添加新的可视化类型

**前端** (`custom-component-registry.tsx`):

```typescript
export type CustomComponentType =
  | "chart"
  | "table"
  | "heatmap"  // 新增
  | "timeline"; // 新增

// 在 CustomComponentRenderer 中添加
case "heatmap":
  return <HeatmapComponent data={config.data} />;

case "timeline":
  return <TimelineComponent data={config.data} />;
```

**后端**:

```python
message = AIMessage(
    content="时间线视图：",
    additional_kwargs={
        "component": {
            "type": "timeline",
            "data": {
                "events": [
                    {"date": "2024-01", "title": "项目启动"},
                    {"date": "2024-03", "title": "完成开发"},
                ]
            }
        }
    }
)
```

## 🚀 迁移指南

### 从 Shadow Root 迁移到组件注册表

**步骤 1：识别组件类型**

分析现有的 shadow root 组件，归类为：

- 图表类（chart）
- 表格类（table）
- 卡片类（card）
- 自定义类（custom）

**步骤 2：提取数据结构**

将 HTML 中的数据提取为 JSON 配置：

```python
# Before
html = f"""
<div class="chart">
    <script>
        echarts.init().setOption({option_json});
    </script>
</div>
"""

# After
config = {
    "type": "chart",
    "data": {"option": option_dict}
}
```

**步骤 3：更新后端代码**

```python
# Before
return Command(custom_events=[{"type": "ui", "content": html}])

# After
return {"messages": [AIMessage(
    content="结果如下：",
    additional_kwargs={"component": config}
)]}
```

**步骤 4：测试验证**

在浏览器中验证：

1. 组件是否正确渲染
2. 样式是否符合预期
3. 交互是否正常工作

## 📈 性能对比

| 方案          | 初始加载   | 渲染速度   | 内存占用   | 维护成本   |
| ------------- | ---------- | ---------- | ---------- | ---------- |
| 组件注册表    | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐   |
| Markdown 扩展 | ⭐⭐⭐⭐   | ⭐⭐⭐⭐   | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐ |
| Shadow Root   | ⭐⭐       | ⭐⭐⭐     | ⭐⭐⭐     | ⭐⭐       |

## 🎓 最佳实践

### 1. 组件类型设计

```typescript
// ✅ 好的设计：明确的类型和数据结构
interface ChartComponentConfig {
  type: "chart";
  data: {
    option: EChartsOption;
    theme?: string;
    loading?: boolean;
  };
}

// ❌ 不好的设计：模糊的类型
interface ComponentConfig {
  type: string;
  data: any;
}
```

### 2. 错误处理

```typescript
// ✅ 提供友好的错误提示
try {
  const config = JSON.parse(configString);
  return <ChartPreview config={config} />;
} catch (error) {
  return (
    <div className="error-box">
      <p>图表配置解析失败</p>
      <pre>{configString}</pre>
    </div>
  );
}
```

### 3. 加载状态

```typescript
// ✅ 使用 Suspense 和 loading 状态
const ChartPreview = dynamic(() => import("./chart-preview"), {
  loading: () => <ChartSkeleton />,
  ssr: false,
});
```

### 4. 类型安全

```python
# ✅ 使用 TypedDict 定义配置结构
from typing_extensions import TypedDict

class ChartComponentConfig(TypedDict):
    type: Literal["chart"]
    data: dict

def create_chart_message(option: dict) -> AIMessage:
    config: ChartComponentConfig = {
        "type": "chart",
        "data": {"option": option}
    }
    return AIMessage(
        content="图表已生成",
        additional_kwargs={"component": config}
    )
```

## 📚 相关文档

- [前端组件集成指南](custom-components.md)
- [后端示例代码](src/react_agent/chart_frontend_component_example.py)
- [组件注册表源码](agent-chat-ui/src/components/thread/messages/custom-component-registry.tsx)
- [Markdown 扩展源码](agent-chat-ui/src/components/thread/messages/markdown-extensions.tsx)

## 🤔 常见问题

**Q: 三种方案可以同时使用吗？**
A: 可以！它们是互补的，会同时生效。

**Q: 如何选择合适的方案？**
A:

- 标准组件 → 组件注册表
- 文本嵌入 → Markdown 扩展
- 完全动态 → Shadow Root

**Q: 性能会有影响吗？**
A: 组件注册表和 Markdown 扩展的性能优于 Shadow Root。

**Q: 如何调试？**
A: 在浏览器控制台查看 `stream.messages` 和 `stream.values.ui`。

**Q: 支持自定义主题吗？**
A: 支持！可以在组件配置中传递 `theme` 或 `className`。

## 🎉 总结

**推荐使用方案 1（组件注册表）**，因为它提供了：

- ✅ 最佳的开发体验
- ✅ 良好的性能
- ✅ 易于维护和扩展
- ✅ 类型安全

对于简单的场景，**方案 2（Markdown 扩展）** 也是不错的选择。

**方案 3（Shadow Root）** 仅在需要完全动态组件时使用。
