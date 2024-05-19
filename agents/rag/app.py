import langchain
import streamlit as st
from langchain.globals import set_llm_cache
from chain import *


@st.cache_resource
def create_llm_cache():
    return fetch_llm_cache()


@st.cache_resource
def create_vector_db(arxiv_query: str, num_docs: int) -> FAISS:
    docs = read_docs(arxiv_query, num_docs)
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
    if not st.session_state["query_llm"]:
        st.warning("Could not find query_llm to clear cache of")
    query_llm = st.session_state["query_llm"]
    query_llm_string = query_llm._get_llm_string()
    langchain.llm_cache.clear(llm_string=query_llm_string)

    if not st.session_state["llm"]:
        st.warning("Could not find llm to clear cache of")
    llm = st.session_state["llm"]
    llm_string = llm._get_llm_string()
    langchain.llm_cache.clear(llm_string=llm_string)


def setup_ui():
    set_llm_cache(create_llm_cache())
    # get prompt template
    # init memory
    memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")

    # Define Prompt Template for New Question Generation using History
    CONVERSATION_TEMPLATE = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:
    """

    # DEFINING READER Prompt Template
    CITATIONS_TEMPLATE = """You are an AI assistant for answering questions about technical topics.
    You are given the following extracted parts of long documents and a question. Provide a conversational answer.
    Use the context as a source of information, but be sure to answer the question directly. You're
    job is to provide the user a helpful summary of the information in the context if it applies to the question.
    If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.

    Question: {question}
    =========
    {context}
    =========
    Answer:
    """

    query_llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        temperature=0.0,
        model_kwargs={"stop": ["<|eot_id|>"]},
    )
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        temperature=0.2,
        max_tokens=1024,
        model_kwargs={"stop": ["<|eot_id|>"]},
    )

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

    with st.sidebar:
        st.write("## LLM Settings")
        ##st.write("### Prompt") TODO make possible to change prompt
        st.write("Change these before you run the app!")
        st.slider("Number of Tokens", 100, 8000, 400, key="max_tokens")

        st.write("## Retrieval Settings")
        st.write("Feel free to change these anytime")
        st.slider("Number of Context Documents", 2, 20, 2, key="num_context_docs")
        st.slider("Distance Threshold", .1, .9, .5, key="distance_threshold", step=.1)

        st.write("## App Settings")
        st.button("Clear Chat", key="clear_chat", on_click=lambda: st.session_state['messages'].clear())
        st.button("Clear Cache", key="clear_cache", on_click=clear_cache)
        st.button("New Conversation", key="reset", on_click=reset_app)

    col1, col2 = st.columns(2)
    with col1:
        st.title("Arxiv ChatGuru")
        st.write("**Put in a topic area and a question within that area to get an answer!**")
        topic = st.text_input("Topic Area", key="arxiv_topic")
        papers = st.number_input("Number of Papers", key="num_papers", value=10, min_value=1, max_value=50, step=2)
    with col2:
        st.image("./assets/arxivguru_crop.png")



    if st.button("Chat!"):
        if is_updated(topic):
            st.session_state['previous_topic'] = topic
            with st.spinner("Loading information from Arxiv to answer your question..."):
                create_arxiv_index(st.session_state['arxiv_topic'], st.session_state['num_papers'], prompt)

    arxiv_db = st.session_state['arxiv_db']
    if st.session_state["llm"] is None:
        tokens = st.session_state["max_tokens"]
        st.session_state["llm"] = get_llm(max_tokens=tokens)
    try:
        standalone_question_chain = get_conversation_chain(query_llm, conversation_template)
        reader, answer = get_reader_chain(llm, reader_template, retriever)
        chain = loaded_memory | standalone_question_chain | reader | answer
        st.session_state['chain'] = chain
    except AttributeError:
        st.info("Please enter a topic area")
        st.stop()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("What do you want to know about this topic?"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant", avatar="./assets/arxivguru_crop.png"):
            message_placeholder = st.empty()
            st.session_state['context'], st.session_state['response'] = [], ""
            chain = st.session_state['chain']

            result = chain({"query": query})
            st.markdown(result["result"])
            st.session_state['context'], st.session_state['response'] = result['source_documents'], result['result']
            if st.session_state['context']:
                with st.expander("Context"):
                    context = defaultdict(list)
                    for doc in st.session_state['context']:
                        context[doc.metadata['Title']].append(doc)
                    for i, doc_tuple in enumerate(context.items(), 1):
                        title, doc_list = doc_tuple[0], doc_tuple[1]
                        st.write(f"{i}. **{title}**")
                        for context_num, doc in enumerate(doc_list, 1):
                            st.write(f" - **Context {context_num}**: {doc.page_content}")

            st.session_state.messages.append({"role": "assistant", "content": st.session_state['response']})


setup_ui()