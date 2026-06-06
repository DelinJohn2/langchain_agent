import asyncio
import os

import streamlit as st
from langchain_core.messages import AIMessageChunk
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

st.set_page_config(page_title="LangChain Docs Helper", page_icon="📚")


def secret(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


LLM_PROVIDER = secret("LLM_PROVIDER", "openai").lower()
LLM_MODEL = secret("LLM_MODEL", "gpt-5.4-mini")
OPENAI_API_KEY = secret("OPENAI_API_KEY")
ANTHROPIC_API_KEY = secret("ANTHROPIC_API_KEY")
CONTEXT7_API_KEY = secret("CONTEXT7_API_KEY")  # optional, raises rate limits

SYSTEM_PROMPT = (
    "You are a LangChain expert assistant. Answer ANY question about LangChain, "
    "LangGraph, LangChain integrations, and the wider ecosystem. Before answering, "
    "use the Context7 tools to look up the latest LangChain documentation (resolve the "
    "library id, then fetch the relevant docs). Recommend the specific module/class, "
    "show minimal `import` + usage snippets and the pip install line where helpful, and "
    "explain concepts clearly. Ground every answer in the fetched docs and be concise."
)


def make_model():
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=LLM_MODEL, api_key=ANTHROPIC_API_KEY, streaming=True
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY, streaming=True)


@st.cache_resource
def build_agent():
    headers = {"CONTEXT7_API_KEY": CONTEXT7_API_KEY} if CONTEXT7_API_KEY else {}
    client = MultiServerMCPClient(
        {
            "context7": {
                "transport": "streamable_http",
                "url": "https://mcp.context7.com/mcp",
                "headers": headers,
            }
        }
    )
    tools = asyncio.run(client.get_tools())
    return create_react_agent(make_model(), tools, prompt=SYSTEM_PROMPT)


def text_of(content):
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return content or ""


def stream_answer(history):
    """Drive the async agent stream synchronously, yielding answer tokens."""
    loop = asyncio.new_event_loop()
    agen = agent.astream({"messages": history}, stream_mode="messages")
    try:
        while True:
            try:
                chunk, _meta = loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                break
            if isinstance(chunk, AIMessageChunk):
                token = text_of(chunk.content)
                if token:
                    yield token
    finally:
        loop.run_until_complete(agen.aclose())
        loop.close()


st.title("📚 LangChain Documentation Helper")
st.caption("Ask anything about LangChain — answers are grounded in live docs via Context7.")

provider_key = OPENAI_API_KEY if LLM_PROVIDER != "anthropic" else ANTHROPIC_API_KEY
if not provider_key:
    st.error(
        f"Set the API key for `{LLM_PROVIDER}` in Streamlit secrets or your environment."
    )
    st.stop()

agent = build_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask anything about LangChain..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [(m["role"], m["content"]) for m in st.session_state.messages]
    with st.chat_message("assistant"):
        answer = st.write_stream(stream_answer(history))

    st.session_state.messages.append({"role": "assistant", "content": answer})
