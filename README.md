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

启动时自动从 GitHub 拉取 [ninehills/pi-commandcode-provider](https://github.com/ninehills/pi-commandcode-provider) 仓库里最新的 `models.json`，包含 18+ 个模型：Claude、GPT、DeepSeek、Kimi、GLM、MiniMax、Qwen、Gemini、Step 等。

- 成功则解析并**缓存到本地 `models.json`**，方便下次离线启动
- 拉取失败则回退到本地缓存（如果存在）
- 两者都失败则返回空列表，服务仍可启动

```bash
# 手动刷新模型列表（重启服务即可）
curl http://localhost:3000/v1/models
```
