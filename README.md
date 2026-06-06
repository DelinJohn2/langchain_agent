# 📚 LangChain Documentation Helper

A bare-minimum Streamlit app that answers "which LangChain library do I use for X?"
(pypdf, FAISS, Chroma, OpenAI, ...). A Claude agent reads the **live** LangChain docs
through the hosted **Context7 MCP** endpoint before answering — no Node/npx subprocess,
so it deploys cleanly to Streamlit Cloud.

## Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then add your key
streamlit run app.py
```

Or use an env var instead of secrets: `export ANTHROPIC_API_KEY=sk-ant-...`

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repo.
2. On https://share.streamlit.io → **New app**, point it at `app.py`.
3. **Settings → Secrets**: add `ANTHROPIC_API_KEY` (see `secrets.toml.example`).

## Config (secrets or env vars)

| Key | Required | Default | Purpose |
|-----|----------|---------|---------|
| `LLM_PROVIDER` | ❌ | `openai` | `openai` or `anthropic` |
| `LLM_MODEL` | ❌ | `gpt-5.4-mini` | Model id |
| `OPENAI_API_KEY` | ✅ (openai) | — | OpenAI key |
| `ANTHROPIC_API_KEY` | ✅ (anthropic) | — | Claude key |
| `CONTEXT7_API_KEY` | ❌ | — | Higher Context7 rate limits |

Answers stream token-by-token, and the agent handles any LangChain / LangGraph question.
# langchain_agent
