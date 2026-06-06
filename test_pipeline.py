"""End-to-end smoke test: Context7 MCP -> agent tool loop -> OpenAI -> streaming."""
import asyncio
import re

import tomllib
from langchain_core.messages import AIMessageChunk
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

cfg = tomllib.load(open(".streamlit/secrets.toml", "rb"))
MODEL = cfg["LLM_MODEL"]
print(f"Provider={cfg['LLM_PROVIDER']} Model={MODEL}")

SYSTEM_PROMPT = (
    "You are a LangChain expert. Use the Context7 tools to look up the latest "
    "LangChain docs before answering, then recommend the specific module/class "
    "with a minimal import + pip line. Be concise."
)


async def main():
    client = MultiServerMCPClient(
        {"context7": {"transport": "streamable_http",
                      "url": "https://mcp.context7.com/mcp", "headers": {}}}
    )
    tools = await client.get_tools()
    print("MCP tools:", [t.name for t in tools])

    model = ChatOpenAI(model=MODEL, api_key=cfg["OPENAI_API_KEY"], streaming=True)
    agent = create_react_agent(model, tools, prompt=SYSTEM_PROMPT)

    q = "Which LangChain library can I use to read a PDF, and which for FAISS vector store?"
    print(f"\nQ: {q}\nA: ", end="", flush=True)

    tools_called = []
    out = []
    async for chunk, meta in agent.astream({"messages": [("user", q)]},
                                           stream_mode="messages"):
        if isinstance(chunk, AIMessageChunk):
            if chunk.tool_calls:
                tools_called += [tc["name"] for tc in chunk.tool_calls]
            txt = chunk.content if isinstance(chunk.content, str) else "".join(
                p.get("text", "") for p in chunk.content if isinstance(p, dict))
            if txt:
                out.append(txt)
                print(txt, end="", flush=True)

    answer = "".join(out)
    print("\n\n--- CHECKS ---")
    print("tools invoked:", tools_called)
    print("answer length:", len(answer))
    hits = [w for w in ["pypdf", "PyPDF", "FAISS", "faiss"] if w in answer]
    print("keyword hits:", hits)
    assert tools_called, "no Context7 tool was called"
    assert len(answer) > 80, "answer too short / empty stream"
    print("\nPIPELINE OK ✅")


asyncio.run(main())
