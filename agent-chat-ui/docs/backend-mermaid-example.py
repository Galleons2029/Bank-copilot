"""
后端集成 Mermaid 图表示例

展示如何在 LangGraph/LangChain 后端返回包含 Mermaid 图表的消息
"""

from langchain_core.messages import AIMessage
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


# ===== 示例 1: 在普通消息中返回 Mermaid 图表 =====


def create_flowchart_response():
    """
    返回包含流程图的消息
    """
    content = """
我来解释一下系统的工作流程：

```mermaid
graph TD
    A[用户提问] --> B{需要检索?}
    B -->|是| C[执行RAG检索]
    B -->|否| D[直接生成]
    C --> E[合并上下文]
    E --> F[生成答案]
    D --> F
    F --> G[返回用户]
```

这个流程图展示了系统如何智能地决定是否需要检索相关信息。
"""

    return AIMessage(content=content)


# ===== 示例 2: 序列图展示交互过程 =====


def create_sequence_diagram_response():
    """
    返回包含序列图的消息
    """
    content = """
让我用序列图来说明整个交互过程：

```mermaid
sequenceDiagram
    participant User as 用户
    participant UI as 前端界面
    participant API as API服务
    participant Agent as LangGraph代理
    participant RAG as RAG系统
    participant LLM as 大语言模型
    
    User->>UI: 发送问题
    UI->>API: POST /chat
    API->>Agent: 处理请求
    Agent->>RAG: 检索相关文档
    RAG-->>Agent: 返回文档片段
    Agent->>LLM: 生成答案
    LLM-->>Agent: 返回答案
    Agent-->>API: 返回结果
    API-->>UI: 返回响应
    UI-->>User: 显示答案
```

整个过程通常在 2-3 秒内完成。
"""

    return AIMessage(content=content)


# ===== 示例 3: 在 LangGraph 节点中使用 =====


class State(TypedDict):
    """状态定义"""

    messages: Annotated[list, add_messages]
    need_diagram: bool


def analyze_query(state: State) -> State:
    """分析用户查询"""
    last_message = state["messages"][-1]

    # 判断是否需要图表说明
    need_diagram = "流程" in last_message.content or "架构" in last_message.content

    return {"messages": state["messages"], "need_diagram": need_diagram}


def generate_response(state: State) -> State:
    """生成响应"""

    if state.get("need_diagram"):
        # 返回包含图表的响应
        response = """
这是我们的系统架构：

```mermaid
graph LR
    A[用户界面] --> B[API网关]
    B --> C[认证服务]
    B --> D[业务服务]
    D --> E[数据库]
    D --> F[缓存Redis]
    D --> G[消息队列]
```

主要组件说明：
- **API网关**: 统一入口，负责路由和限流
- **认证服务**: 处理用户认证和授权
- **业务服务**: 核心业务逻辑
- **数据库**: 持久化存储
- **缓存**: 提高读取性能
- **消息队列**: 异步任务处理
"""
    else:
        response = "这是一个普通的回答，不需要图表说明。"

    return {"messages": [AIMessage(content=response)]}


# ===== 示例 4: 根据不同场景返回不同图表 =====


def create_diagram_by_scenario(scenario: str) -> AIMessage:
    """
    根据场景返回不同类型的图表
    """

    diagrams = {
        "architecture": """
系统架构如下：

```mermaid
graph TB
    subgraph 前端层
        A[Web应用]
        B[移动应用]
    end
    subgraph 应用层
        C[API服务]
        D[WebSocket服务]
    end
    subgraph 业务层
        E[LangGraph代理]
        F[RAG系统]
    end
    subgraph 数据层
        G[PostgreSQL]
        H[向量数据库]
        I[Redis]
    end
    
    A --> C
    B --> C
    A --> D
    B --> D
    C --> E
    D --> E
    E --> F
    F --> H
    E --> G
    E --> I
```
""",
        "workflow": """
工作流程说明：

```mermaid
stateDiagram-v2
    [*] --> 接收请求
    接收请求 --> 解析意图
    解析意图 --> 需要工具?
    需要工具? --> 调用工具: 是
    需要工具? --> 直接回答: 否
    调用工具 --> 处理结果
    处理结果 --> 生成回复
    直接回答 --> 生成回复
    生成回复 --> [*]
```
""",
        "timeline": """
项目时间规划：

```mermaid
gantt
    title 项目开发计划
    dateFormat YYYY-MM-DD
    
    section 需求阶段
    需求调研           :done, req1, 2024-01-01, 5d
    需求分析           :done, req2, after req1, 3d
    
    section 设计阶段
    系统设计           :active, des1, 2024-01-09, 7d
    UI设计            :active, des2, 2024-01-09, 7d
    
    section 开发阶段
    后端开发           :dev1, 2024-01-16, 14d
    前端开发           :dev2, 2024-01-16, 14d
    
    section 测试阶段
    单元测试           :test1, 2024-01-30, 5d
    集成测试           :test2, after test1, 3d
    
    section 上线阶段
    预发布            :deploy1, 2024-02-07, 2d
    正式发布           :deploy2, after deploy1, 1d
```
""",
        "data_model": """
数据模型关系：

```mermaid
erDiagram
    USER ||--o{ CONVERSATION : has
    USER {
        int id PK
        string username
        string email
        datetime created_at
    }
    
    CONVERSATION ||--|{ MESSAGE : contains
    CONVERSATION {
        int id PK
        int user_id FK
        string title
        datetime created_at
    }
    
    MESSAGE {
        int id PK
        int conversation_id FK
        string role
        text content
        datetime created_at
    }
    
    MESSAGE ||--o{ TOOL_CALL : may_have
    TOOL_CALL {
        int id PK
        int message_id FK
        string tool_name
        json arguments
        json result
    }
```
""",
    }

    content = diagrams.get(scenario, "抱歉，我不知道如何为这个场景生成图表。")

    return AIMessage(content=content)


# ===== 示例 5: 与工具调用结合 =====


def analyze_data_with_visualization(data: dict) -> AIMessage:
    """
    分析数据并返回可视化结果
    """

    # 模拟数据分析
    total = sum(data.values())
    percentages = {k: (v / total * 100) for k, v in data.items()}

    # 生成饼图
    pie_chart = "pie title 数据分布\n"
    for key, value in percentages.items():
        pie_chart += f'    "{key}" : {value:.1f}\n'

    content = f"""
数据分析完成！以下是可视化结果：

```mermaid
{pie_chart}
```

**数据说明**：
"""

    for key, value in data.items():
        percentage = percentages[key]
        content += f"\n- **{key}**: {value} ({percentage:.1f}%)"

    content += f"\n\n总计: {total}"

    return AIMessage(content=content)


# ===== 示例 6: 动态生成流程图 =====


def generate_process_flowchart(steps: list[dict]) -> AIMessage:
    """
    根据步骤列表动态生成流程图

    Args:
        steps: [{"id": "A", "label": "步骤1", "next": "B"}, ...]
    """

    # 构建 Mermaid 语法
    mermaid_code = "graph TD\n"

    for step in steps:
        step_id = step["id"]
        label = step["label"]
        next_step = step.get("next")

        if next_step:
            if isinstance(next_step, list):
                # 多个分支
                for branch in next_step:
                    condition = branch.get("condition", "")
                    target = branch["target"]
                    arrow = f"|{condition}|" if condition else ""
                    mermaid_code += f"    {step_id}[{label}] -->{arrow} {target}\n"
            else:
                # 单个分支
                mermaid_code += f"    {step_id}[{label}] --> {next_step}\n"
        else:
            # 结束节点
            mermaid_code += f"    {step_id}[{label}]\n"

    content = f"""
流程说明：

```mermaid
{mermaid_code}
```

这个流程包含 {len(steps)} 个步骤。
"""

    return AIMessage(content=content)


# ===== 使用示例 =====

if __name__ == "__main__":
    # 示例 1: 简单流程图
    print("=" * 50)
    print("示例 1: 流程图")
    print("=" * 50)
    msg1 = create_flowchart_response()
    print(msg1.content)

    # 示例 2: 序列图
    print("\n" + "=" * 50)
    print("示例 2: 序列图")
    print("=" * 50)
    msg2 = create_sequence_diagram_response()
    print(msg2.content)

    # 示例 3: 根据场景生成
    print("\n" + "=" * 50)
    print("示例 3: 系统架构图")
    print("=" * 50)
    msg3 = create_diagram_by_scenario("architecture")
    print(msg3.content)

    # 示例 4: 数据分析可视化
    print("\n" + "=" * 50)
    print("示例 4: 数据分析可视化")
    print("=" * 50)
    sample_data = {"Python": 35, "JavaScript": 25, "TypeScript": 20, "Go": 15, "其他": 5}
    msg4 = analyze_data_with_visualization(sample_data)
    print(msg4.content)

    # 示例 5: 动态流程图
    print("\n" + "=" * 50)
    print("示例 5: 动态流程图")
    print("=" * 50)
    process_steps = [
        {"id": "A", "label": "开始", "next": "B"},
        {"id": "B", "label": "接收请求", "next": "C"},
        {"id": "C", "label": "判断类型", "next": [{"condition": "查询", "target": "D"}, {"condition": "分析", "target": "E"}]},
        {"id": "D", "label": "执行查询", "next": "F"},
        {"id": "E", "label": "执行分析", "next": "F"},
        {"id": "F", "label": "返回结果", "next": None},
    ]
    msg5 = generate_process_flowchart(process_steps)
    print(msg5.content)

    print("\n" + "=" * 50)
    print("所有示例完成！")
    print("=" * 50)
