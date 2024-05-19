import langchain
import streamlit as st
from langchain.globals import set_llm_cache
from chain import *


@st.cache_resource
def create_llm_cache():
    return fetch_llm_cache()


@st.cache_resource
def create_vector_db(arxiv_query: str) -> FAISS:
    docs = read_docs(arxiv_query)
    db = get_vector_db(docs)
    st.session_state["db"] = db
    return db


def reset_app():
    st.session_state["query"] = ""
    st.session_state["topic"] = ""
    st.session_state["messages"].clear()

    db = st.session_state["db"]
    if db is not None:
        ...
        st.session_state["db"] = None


def clear_cache():
    if not st.session_state["llm"]:
        st.warning("Could not find llm to clear cache of")
    llm = st.session_state["llm"]
    llm_string = llm._get_llm_string()
    langchain.llm_cache.clear(llm_string=llm_string)


def start_app():
    set_llm_cache(create_llm_cache())
    # get prompt template

    # Defining default values
    default_question = ""
    default_answer = ""
    defaults = {
        "response": {"choices": [{"text": default_answer}]},
        "question": default_question,
        "context": [],
        "chain": None,
        "arxiv_topic": "",
        "arxiv_query": "",
        "db": None,
        "llm": None,
        "messages": [],
    }

    # Checking if keys exist in session state, if not, initializing them
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
