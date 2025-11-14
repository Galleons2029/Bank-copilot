# UI 组件构建

可独立构建的 React UI 组件，用于集成到 LangGraph Python 后端。

## 快速开始

### 构建

```bash
# 推荐（从项目根目录）
make build_ui

# 或从 agent-chat-ui 目录
pnpm run build:ui
node build-ui.mjs
```

### 输出

```
ui/dist/
├── chart-preview.js        # 98.6 KB
├── chart-preview.css       # 36.5 KB
└── *.map                   # Source Maps
```

## 使用示例

```python
from langgraph_api.js import load_ui

chart_ui = load_ui("chart-preview", path="agent-chat-ui/ui/dist")

def my_agent_node(state):
    charts = [{
        "id": "chart-1",
        "title": "销售数据",
        "data": base64_image_string,
        "mime_type": "image/png"
    }]
    return {"ui": chart_ui(charts=charts)}
```

## 技术栈

- **构建**: esbuild + Tailwind CSS v4
- **模块**: ESM (ES2020)
- **外部依赖**: react, react-dom, lucide-react

## 组件特性

### chart-preview

- ✅ 响应式图表卡片（水平滚动）
- ✅ 悬停显示操作按钮（Shadow DOM 兼容）
- ✅ 居中放大预览（平滑缩放动画）
- ✅ 一键下载图表
- ✅ 完整无障碍支持

## 开发

1. 修改 `ui/chart-preview.tsx`
2. 构建 `make build_ui`
3. 测试 `make dev_ui`

## 文件结构

```
ui/
├── chart-preview.tsx       # 组件主文件
├── chart-preview.types.ts  # 类型定义
├── style.css               # Tailwind 入口
└── dist/                   # 构建输出
```

## 构建配置

位于 `build-ui.mjs`：TypeScript 编译、JSX 转换、Tailwind 处理、代码压缩、Source Maps。
