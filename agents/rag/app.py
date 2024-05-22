from collections import defaultdict
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
        clear_cache()
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
    memory = ConversationBufferMemory(
        return_messages=True, output_key="answer", input_key="question"
    )

    # Define Prompt Template for New Question Generation using History
    CONVERSATION_TEMPLATE = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:
    """

    # DEFINING READER Prompt Template
    READER_TEMPLATE = """You are an AI assistant for answering questions about technical topics.
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
        "query_llm": None,
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

        st.write("## App Settings")
        st.button("Clear Chat", key="clear_chat", on_click=lambda: st.session_state['messages'].clear())
        st.button("Clear Cache", key="clear_cache", on_click=clear_cache)
        st.button("New Conversation", key="reset", on_click=reset_app)

    st.title("Arxiv Chat")
    st.write("**Put in a topic area and a question within that area to get an answer!**")
    topic = st.text_input("Topic Area", key="arxiv_topic")
    num_docs = st.number_input(
        "Number of Pages", key="num_docs", value=10, min_value=1, max_value=50, step=2
    )

    if st.button("Chat!"):
        # if is_updated(topic):
        #     st.session_state['previous_topic'] = topic
        with st.spinner("Loading information from Arxiv to answer your question..."):
            create_vector_db(st.session_state['arxiv_topic'], st.session_state['num_docs'])

    db = st.session_state["db"]
    if st.session_state["query_llm"] is None or st.session_state["llm"] is None:
        tokens = st.session_state["max_tokens"]
        st.session_state["query_llm"], st.session_state["llm"] = get_llms(max_tokens=tokens)
    try:
        # DEFINING RETRIEVER
        retriever = db.as_retriever()
        # load the memory to access chat history
        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
        )
        standalone_question_chain = get_conversation_chain(st.session_state["query_llm"], CONVERSATION_TEMPLATE)
        reader, answer = get_reader_chain(st.session_state["llm"], READER_TEMPLATE, retriever)
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

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            st.session_state['context'], st.session_state['response'] = [], ""
            chain = st.session_state['chain']
            inputs = {"question": query}
            full_response = {}
            answer = ""
            for chunk in chain.stream(inputs):
                for key in chunk:
                    if key not in full_response:
                        full_response[key] = chunk[key]
                    else:
                        full_response[key] += chunk[key]
                    if key == 'answer':
                        answer += chunk[key].content
                        with message_placeholder.container():
                            st.markdown(answer)

            # save the current question and answer to memory as chat history
            memory.save_context(inputs, {"answer": answer})
            st.session_state['context'], st.session_state['response'] = full_response['docs'], answer
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
