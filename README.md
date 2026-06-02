# commandcode2api

将 [Command Code](https://commandcode.ai) API 代理为 OpenAI 兼容的 Chat Completion API。

## 特性

- **OpenAI 兼容格式**：支持 `/v1/chat/completions`（流式 + 非流式）和 `/v1/models`。
- **文本 / 推理 / 工具调用**：完整支持 CommandCode 的 `text-delta`、`reasoning-delta`、`tool-call` 事件。
- **标准 SSE 流式返回**：客户端可用任何 OpenAI SDK 直接连接。
- **认证灵活**：从请求 `Authorization` Header 或环境变量 `COMMANDCODE_API_KEY` 读取 API Key。

## 开发环境

本项目使用 Nix flake 管理开发 Shell，已包含 Rust 工具链（rustc、cargo、clippy、rustfmt、rust-analyzer）及编译依赖。

```bash
# 进入 devShell（首次会自动下载）
nix develop

# 如果装了 direnv，进入目录会自动加载
# echo "use flake" > .envrc && direnv allow
```

## 运行

```bash
# 1. 设置你的 CommandCode API Key
export COMMANDCODE_API_KEY="user_..."

# 2. 启动服务
cargo run

# 默认监听 0.0.0.0:3000
# 可通过 PORT 环境变量修改端口
PORT=8080 cargo run
```

## 使用示例

### curl

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Authorization: Bearer $COMMANDCODE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek/deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

### Python (openai SDK)

```python
from openai import OpenAI

client = OpenAI(
    api_key="user_...",          # 你的 CommandCode API Key
    base_url="http://localhost:3000/v1",
)

response = client.chat.completions.create(
    model="deepseek/deepseek-v4-flash",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `COMMANDCODE_API_KEY` | 默认 CommandCode API Key（请求 Header 优先） | - |
| `COMMANDCODE_API_BASE` | CommandCode API 基础地址 | `https://api.commandcode.ai` |
| `PORT` | 监听端口 | `3000` |
| `RUST_LOG` | 日志级别 | `info` |

## 支持的模型

模型列表通过 `scripts/extract-models.js` 从官方 `command-code` npm 包的 `dist/index.mjs` bundle 中提取生成 `models.json`（解决了原 GitHub 仓库 404 的问题）。

- 运行 `node scripts/extract-models.js` 可**更新**本地 `models.json`（基于最新发布的 CLI 版本）。
- 服务启动时优先尝试远程加载（当前上游地址已 404），失败则回退使用本地的 `models.json`。
- 两者都失败则返回空列表，但服务仍可启动（可直接传入 model id 调用上游）。

```bash
# 更新模型列表（然后重启服务）
node scripts/extract-models.js

# 查看当前可用模型
curl http://localhost:3000/v1/models
```

### 当前可用模型（共 26 个）

按系列分组（可直接在 `model` 参数中使用这些 ID）：

**Claude 系列 (Anthropic)**

- `claude-sonnet-4-6`
- `claude-opus-4-8`
- `claude-opus-4-7`
- `claude-haiku-4-5-20251001`

**GPT 系列 (OpenAI)**

- `gpt-5.5`
- `gpt-5.4`
- `gpt-5.4-mini`
- `gpt-5.3-codex`

**DeepSeek**

- `deepseek/deepseek-v4-flash`
- `deepseek/deepseek-v4-pro`

**Gemini (Google)**

- `google/gemini-3.5-flash`
- `google/gemini-3.1-flash-lite`

**Qwen 系列 (Alibaba)**

- `Qwen/Qwen3.7-Max`
- `Qwen/Qwen3.6-Max-Preview`
- `Qwen/Qwen3.6-Plus`

**Kimi 系列 (Moonshot)**

- `moonshotai/Kimi-K2.6`
- `moonshotai/Kimi-K2.5`

**GLM 系列 (Zhipu / Z.ai)**

- `zai-org/GLM-5.1`
- `zai-org/GLM-5`

**MiniMax**

- `MiniMaxAI/MiniMax-M3`
- `MiniMaxAI/MiniMax-M2.7`
- `MiniMaxAI/MiniMax-M2.5`

**Step 系列 (StepFun)**

- `stepfun/Step-3.7-Flash`
- `stepfun/Step-3.5-Flash`

**MiMo 系列 (Xiaomi)**

- `xiaomi/mimo-v2.5-pro`
- `xiaomi/mimo-v2.5`

> **说明**：大部分开放模型由 Command Code 托管提供（`owned_by` 通常为 `command-code`），Claude/GPT 部分可走原厂提供商。实际可用模型和额度取决于你的 CommandCode 账户。列表会随上游更新，请定期执行提取脚本并重启服务。
