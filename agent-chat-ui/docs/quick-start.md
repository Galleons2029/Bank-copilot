# 🚀 前端自定义组件快速开始指南

## 📦 已完成的工作

✅ **前端组件注册表系统** - 完整实现
✅ **配置提取逻辑** - 支持 3 种方式
✅ **测试页面** - 完整的功能演示
✅ **路径别名配置** - tsconfig 和 webpack
✅ **功能测试** - 所有核心功能通过
✅ **真实图表渲染** - ECharts 集成完成 (echarts@6.0.0)

## 🎯 10 分钟快速上手

### 第 1 步：启动前端服务

```bash
cd agent-chat-ui
pnpm install  # 如果还没安装依赖
pnpm dev      # 启动开发服务器
```

访问测试页面：http://localhost:3000/test-components

### 第 2 步：在后端发送带组件的消息

#### Python 代码示例（最简单）

```python
from langchain_core.messages import AIMessage

# 创建带图表组件的消息
message = AIMessage(
    content="数据分析完成，结果如下：",
    additional_kwargs={
        "component": {
            "type": "chart",  # 组件类型
            "data": {
                "option": {  # ECharts 配置
                    "title": {"text": "销售数据"},
                    "xAxis": {
                        "type": "category",
                        "data": ["1月", "2月", "3月"]
                    },
                    "yAxis": {"type": "value"},
                    "series": [{
                        "data": [120, 200, 150],
                        "type": "line"
                    }]
                }
            }
        }
    }
)

return {"messages": [message]}
```

### 第 3 步：前端自动渲染

前端会自动：

1. ✅ 检测 `additional_kwargs.component`
2. ✅ 提取组件配置
3. ✅ 渲染对应组件类型

## 📊 支持的组件类型

### 1. 图表组件 (type: "chart")

```python
{
    "type": "chart",
    "data": {
        "option": {
            # 完整的 ECharts 配置对象
            "xAxis": {...},
            "yAxis": {...},
            "series": [...]
        }
    }
}
```

### 2. 表格组件 (type: "table")

```python
{
    "type": "table",
    "data": {
        "headers": ["列1", "列2", "列3"],
        "rows": [
            ["数据1", "数据2", "数据3"],
            ["数据4", "数据5", "数据6"]
        ]
    }
}
```

### 3. 图片组件 (type: "image")

```python
{
    "type": "image",
    "data": {
        "url": "https://example.com/image.jpg",
        "alt": "图片描述",
        "caption": "图片说明"
    }
}
```

### 4. 文件组件 (type: "file")

```python
{
    "type": "file",
    "data": {
        "name": "report.pdf",
        "size": 2048576,  # 字节
        "url": "https://example.com/download"
    }
}
```

## 🔧 三种触发方式

### 方式 1：additional_kwargs ⭐ 推荐

```python
message = AIMessage(
    content="结果如下：",
    additional_kwargs={
        "component": {
            "type": "chart",
            "data": {...}
        }
    }
)
```

**优点**：

- ✅ 类型安全
- ✅ 结构清晰
- ✅ 易于维护

### 方式 2：Markdown 代码块

````python
message = AIMessage(
    content="""
分析结果：

\```chart
{"option": {...}}
\```
"""
)
````

**优点**：

- ✅ 对后端侵入最小
- ✅ 可在文本中任意位置插入
- ✅ 易于理解

### 方式 3：Tool Calls

```python
message = AIMessage(
    content="生成图表...",
    tool_calls=[{
        "name": "render_chart",
        "args": {"option": {...}},
        "id": "call_123"
    }]
)
```

**优点**：

- ✅ 符合 Agent 工作流
- ✅ 可追踪工具调用
- ✅ 支持多个组件

## 📝 完整示例：修改现有 Agent

### 原代码（使用 shadow root）

```python
from langgraph.types import Command

def my_agent_node(state):
    # 生成 HTML
    html = f"""
    <div id="chart"></div>
    <script>
        // 复杂的 HTML/JS
    </script>
    """

    return Command(
        custom_events=[{
            "type": "ui",
            "content": html
        }]
    )
```

### 新代码（使用组件注册表）✨

```python
from langchain_core.messages import AIMessage

def my_agent_node(state):
    # 只需准备数据配置
    chart_config = {
        "title": {"text": "数据分析"},
        "xAxis": {"data": ["A", "B", "C"]},
        "series": [{"data": [10, 20, 30], "type": "bar"}]
    }

    message = AIMessage(
        content="数据分析完成",
        additional_kwargs={
            "component": {
                "type": "chart",
                "data": {"option": chart_config}
            }
        }
    )

    return {"messages": [message]}
```

**改进点**：

- ✅ 代码量减少 80%
- ✅ 无需生成 HTML
- ✅ 前端完全控制样式
- ✅ 类型安全
- ✅ 易于测试

## 🎨 在实际 Agent 中使用

### 示例：EChart Agent

```python
# src/react_agent/sub_agent/echart_agent.py

from typing import TypedDict
from langchain_core.messages import AIMessage

class State(TypedDict):
    messages: list

def echart_node(state: State):
    """生成图表的节点"""

    # 1. 分析数据（你的业务逻辑）
    data = analyze_data(state["messages"])

    # 2. 构建 ECharts 配置
    chart_config = {
        "title": {"text": data["title"]},
        "xAxis": {
            "type": "category",
            "data": data["categories"]
        },
        "yAxis": {"type": "value"},
        "series": [{
            "name": data["name"],
            "data": data["values"],
            "type": data["chart_type"]  # line, bar, pie
        }]
    }

    # 3. 创建消息
    message = AIMessage(
        content=f"已生成{data['title']}的可视化图表：",
        additional_kwargs={
            "component": {
                "type": "chart",
                "data": {"option": chart_config}
            }
        }
    )

    return {"messages": [message]}

# 在图中使用
graph.add_node("generate_chart", echart_node)
```

## 🧪 测试你的实现

### 1. 使用测试页面

访问 http://localhost:3000/test-components

### 2. 在 Python 中测试

```python
# 创建测试消息
test_message = AIMessage(
    content="测试消息",
    additional_kwargs={
        "component": {
            "type": "table",
            "data": {
                "headers": ["名称", "数值"],
                "rows": [["A", "100"], ["B", "200"]]
            }
        }
    }
)

# 打印验证
print(test_message.additional_kwargs)
```

### 3. 在浏览器中验证

打开浏览器控制台：

```javascript
// 查看消息结构
console.log(stream.messages);

// 查看组件配置
stream.messages.forEach((msg) => {
  if (msg.additional_kwargs?.component) {
    console.log("Found component:", msg.additional_kwargs.component);
  }
});
```

## 📚 更多资源

### 文档

- 📖 [详细集成指南](custom-components.md) - 完整的使用文档
- 📊 [方案对比](frontend-component-solutions.md) - 三种方案详细对比
- 🧪 [测试报告](test-results.md) - 功能测试结果

### 示例代码

- 🐍 [Python 后端示例](src/react_agent/chart_frontend_component_example.py)
- 🎨 [测试页面源码](agent-chat-ui/src/app/test-components/page.tsx)
- 📦 [组件注册表](agent-chat-ui/src/components/thread/messages/custom-component-registry.tsx)

### 在线演示

- 🌐 测试页面: http://localhost:3000/test-components
- 💬 聊天界面: http://localhost:3000

## 🐛 常见问题

### Q: 组件不显示？

**检查清单**：

1. ✅ 前端服务已启动？
2. ✅ 消息结构正确？
3. ✅ `additional_kwargs.component` 字段存在？
4. ✅ `type` 和 `data` 字段正确？

**调试方法**：

```javascript
// 浏览器控制台
console.log(stream.messages);
```

### Q: 如何添加新的组件类型？

编辑 `custom-component-registry.tsx`：

```typescript
// 1. 添加类型定义
export type CustomComponentType =
  | "chart"
  | "table"
  | "my_new_type";  // 新增

// 2. 添加渲染逻辑
case "my_new_type":
  return <MyNewComponent data={config.data} />;
```

### Q: 支持实时更新吗？

支持！组件会随消息流实时渲染：

```python
# 流式返回多个组件
for data_chunk in data_stream:
    yield AIMessage(
        content=f"处理中... {data_chunk['progress']}%",
        additional_kwargs={
            "component": {
                "type": "chart",
                "data": {"option": generate_chart(data_chunk)}
            }
        }
    )
```

## 🎉 开始使用

现在你已经准备好使用前端自定义组件了！

**建议的学习路径**：

1. 访问测试页面查看示例 (5 分钟)
2. 修改一个现有的 Agent 使用新方式 (15 分钟)
3. 测试并观察效果 (10 分钟)

**总用时**：约 30 分钟即可掌握！

---

💡 **提示**: 遇到问题？查看 [测试报告](test-results.md) 或 [详细文档](custom-components.md)
