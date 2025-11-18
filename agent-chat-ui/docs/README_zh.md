# Agent Chat UI 中文指南

Agent Chat UI 是一个基于 Next.js 的聊天界面，能够与任何提供 `messages` 键的 LangGraph 服务交互。本指南为前端仓库提供中文说明，方便快速了解功能与部署方式。

> 🎥 视频教学：[https://youtu.be/lInrwVnZ83o](https://youtu.be/lInrwVnZ83o)


## 📊 自定义组件系统

项目已内置前端自定义组件系统，可在对话中动态渲染多种结构化内容：

- ✅ **图表组件**：基于 ECharts 渲染交互式图表
- ✅ **表格组件**：展示结构化数据
- ✅ **图片组件**：图片预览与展示
- ✅ **文件组件**：文件信息与下载入口
- ✅ **Mermaid 图表**：渲染流程图、序列图、甘特图等

📖 参考文档：
- [docs/README.md](README.md) - 完整文档
- [docs/quick-start.md](quick-start.md) - 快速开始
- [docs/mermaid-integration.md](mermaid-integration.md) - Mermaid 详细指南

🧪 测试页面：`http://localhost:3000/test-components`


## 快速开始

> 💡 不想本地运行？可直接访问部署示例：[agentchat.vercel.app](https://agentchat.vercel.app)

### 1. 获取项目

```bash
npx create-agent-chat-app
# 或者
git clone https://github.com/langchain-ai/agent-chat-ui.git
cd agent-chat-ui
```

### 2. 安装依赖

```bash
pnpm install
```

### 3. 启动开发环境

```bash
pnpm dev
```

默认访问地址为 `http://localhost:3000`。

### 4. Docker Compose

```bash
docker compose up --build
```

首选使用 `docker compose build frontend` 重新构建镜像，以便同步依赖变动。


## 使用方式

应用启动后（或访问线上版本），界面会要求填写：

- **Deployment URL**：LangGraph 服务地址（可为本地或线上）
- **Assistant/Graph ID**：聊天使用的图或助手 ID
- **LangSmith API Key**：调用 LangGraph 线上部署时使用

填写完毕后点击 `Continue` 即可进入聊天界面。


## 环境变量

可通过环境变量绕过首屏表单：

```bash
NEXT_PUBLIC_API_URL=http://localhost:2024
NEXT_PUBLIC_ASSISTANT_ID=agent
```

配置步骤：

1. 将 `.env.example` 复制为 `.env`
2. 填写所需变量值
3. 重启应用

当设置这些变量后，应用会直接使用它们来连接 LangGraph。


## 隐藏聊天信息

可通过以下两种方式控制消息在 UI 中的可见性：

1. **阻止实时流式展示**：在模型配置中添加 `langsmith:nostream` 标签，阻止 UI 通过 `on_chat_model_stream` 事件渲染流式消息。

    ```python
    from langchain_anthropic import ChatAnthropic

    model = ChatAnthropic().with_config(
        config={"tags": ["langsmith:nostream"]}
    )
    ```

    ```typescript
    import { ChatAnthropic } from "@langchain/anthropic";

    const model = new ChatAnthropic().withConfig({
      tags: ["langsmith:nostream"],
    });
    ```

2. **完全隐藏消息**：在将消息写入图状态前，为 `id` 添加 `do-not-render-` 前缀，并在模型配置中添加 `langsmith:do-not-render` 标签。UI 会过滤掉这些消息。

    ```python
    result = model.invoke([messages])
    result.id = f"do-not-render-{result.id}"
    return {"messages": [result]}
    ```

    ```typescript
    const result = await model.invoke([messages]);
    result.id = `do-not-render-${result.id}`;
    return { messages: [result] };
    ```


## 渲染 Artifact

Agent Chat UI 支持在聊天右侧面板渲染 Artifact。可以通过 `thread.meta.artifact` 获取上下文：

```tsx
export function useArtifact<TContext = Record<string, unknown>>() {
  type Component = (props: {
    children: React.ReactNode;
    title?: React.ReactNode;
  }) => React.ReactNode;

  type Context = TContext | undefined;

  type Bag = {
    open: boolean;
    setOpen: (value: boolean | ((prev: boolean) => boolean)) => void;

    context: Context;
    setContext: (value: Context | ((prev: Context) => Context)) => void;
  };

  const thread = useStreamContext<
    { messages: Message[]; ui: UIMessage[] },
    { MetaType: { artifact: [Component, Bag] } }
  >();

  return thread.meta?.artifact;
}
```

然后使用 `useArtifact` hook 返回的 `Artifact` 组件进行渲染：

```tsx
import { useArtifact } from "../utils/use-artifact";
import { LoaderIcon } from "lucide-react";

export function Writer(props: {
  title?: string;
  content?: string;
  description?: string;
}) {
  const [Artifact, { open, setOpen }] = useArtifact();

  return (
    <>
      <div
        onClick={() => setOpen(!open)}
        className="cursor-pointer rounded-lg border p-4"
      >
        <p className="font-medium">{props.title}</p>
        <p className="text-sm text-gray-500">{props.description}</p>
      </div>

      <Artifact title={props.title}>
        <p className="p-4 whitespace-pre-wrap">{props.content}</p>
      </Artifact>
    </>
  );
}
```


## 生产部署指南

默认情况下，Agent Chat UI 以本地开发为目标，直接在客户端连接到 LangGraph 服务，需每位用户提供自己的 LangSmith API Key。若要用于生产环境，需要改造请求认证流程。

### 方案一：API Passthrough（快速接入）

使用 [langgraph-nextjs-api-passthrough](https://github.com/langchain-ai/langgraph-nextjs-api-passthrough) 可以快速搭建代理 API，并自动为请求注入 LangSmith API Key。本仓库已包含所需代码，只需设置环境变量：

```bash
NEXT_PUBLIC_ASSISTANT_ID="agent"
LANGGRAPH_API_URL="https://my-agent.default.us.langgraph.app"
NEXT_PUBLIC_API_URL="https://my-website.com/api"
LANGSMITH_API_KEY="lsv2_..."
```

- `NEXT_PUBLIC_ASSISTANT_ID`：对话使用的助手/图 ID，需保留 `NEXT_PUBLIC_` 前缀。
- `LANGGRAPH_API_URL`：LangGraph 部署地址。
- `NEXT_PUBLIC_API_URL`：站点地址 + `/api`，供前端访问代理。
- `LANGSMITH_API_KEY`：LangSmith API Key，由代理在服务端注入，不应加 `NEXT_PUBLIC_` 前缀。

更多细节参阅 [LangGraph Next.js API Passthrough](https://www.npmjs.com/package/langgraph-nextjs-api-passthrough) 文档。

### 方案二：自定义认证（高级）

通过 LangGraph 的自定义认证，可允许客户端在无 LangSmith API Key 的情况下安全访问，同时可配置细粒度的访问控制。请参阅 LangGraph 文档：

- [Python 自定义认证](https://langchain-ai.github.io/langgraph/tutorials/auth/getting_started/)
- [TypeScript 自定义认证](https://langchain-ai.github.io/langgraphjs/how-tos/auth/custom_auth/)

部署完成后需在前端做以下调整：

1. 在前端补充请求逻辑，获取并注入部署端返回的认证 Token。
2. 将 `NEXT_PUBLIC_API_URL` 设置为生产 LangGraph 部署地址。
3. 将 `NEXT_PUBLIC_ASSISTANT_ID` 设置为对应助手 ID。
4. 修改 [`useTypedStream`](../src/providers/Stream.tsx)（`useStream` 的扩展）以在 `defaultHeaders` 中携带 Token：

    ```tsx
    const streamValue = useTypedStream({
      apiUrl: process.env.NEXT_PUBLIC_API_URL,
      assistantId: process.env.NEXT_PUBLIC_ASSISTANT_ID,
      defaultHeaders: {
        Authentication: `Bearer ${addYourTokenHere}`,
      },
    });
    ```


## 反馈

如在中文指南中发现缺漏，欢迎在仓库中提交 Issue 或 PR 共同完善。***
