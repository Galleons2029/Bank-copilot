# 🔧 图片组件显示问题修复

## 问题描述

后端 `echart_agent` 通过 `additional_kwargs` 发送的图片数据无法在前端显示。

**后端格式：**

```python
AIMessage(
    additional_kwargs={
        "type": "image",
        "data": {
            "layout": "carousel",
            "images": [{"url": "...", "alt": "图表 1"}]
        }
    }
)
```

## 根本原因

前端只检查 `additional_kwargs.component`，而 `echart_agent` 直接传递 `type` 和 `data`。

## 解决方案

### 修改前端提取逻辑

**文件：** `custom-component-registry.tsx`

```typescript
export function extractComponentConfig(message: Message) {
  const additionalKwargs = (message as any).additional_kwargs;

  // 标准格式
  if (additionalKwargs?.component) {
    return additionalKwargs.component;
  }

  // echart_agent 格式（新增）
  if (additionalKwargs?.type && additionalKwargs?.data) {
    return {
      type: additionalKwargs.type,
      data: additionalKwargs.data,
    };
  }

  // ... 其他方式
}
```

## 支持的格式

### 格式 1：标准格式

```python
additional_kwargs={
    "component": {
        "type": "image",
        "data": { ... }
    }
}
```

### 格式 2：简化格式（echart_agent）

```python
additional_kwargs={
    "type": "image",
    "data": { ... }
}
```

## 优化改进

### 1. 轮播布局尺寸优化

- 最大宽度：768px（`max-w-3xl`）
- 图片高度：500px（`max-h-[500px]`）
- 居中显示：`mx-auto`
- 保持比例：`object-contain`

### 2. 网格布局测试

- 4张图片网格展示
- 每张图片独立 caption
- 适合对比查看

## 测试

访问 http://localhost:3000/test-components 查看示例

## 相关文件

- ✅ `custom-component-registry.tsx` - 组件提取逻辑
- ✅ `test-components/page.tsx` - 测试页面
- ✅ `custom-components.md` - 使用文档
- ✅ `echart_agent.py` - 后端实现
