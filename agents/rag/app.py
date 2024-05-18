""" Refer to https://langchain114.com/docs/expression_language/cookbook/retrieval
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from operator import itemgetter
from termcolor import colored
from langchain.memory import ConversationBufferMemory
from langchain.schema import format_document
from langchain.schema.messages import get_buffer_string
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.document_loaders import ArxivLoader, PyPDFLoader

load_dotenv()

os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def read_pdf(file_path):
    pdf_search = Path(file_path).glob("*.pdf")
    pdf_files = [str(file.absolute()) for file in pdf_search]
    print("Total PDF files", len(pdf_files))
    pages = []
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        pages.extend(loader.load_and_split())
    return pages


def read_docs(arxiv_query: str):
    raw_docs = ArxivLoader(query=arxiv_query,
                       load_max_docs=10,
                       load_all_available_meta=True).load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    docs = text_splitter.split_documents(raw_docs)
    print("Total arxiv pages:", len(docs))
    return docs


def get_conversation_chain(llm, template):
    prompt = PromptTemplate.from_template(template=template)

    llm_chain = prompt | llm

    standalone_question_chain = (
        {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | llm_chain
        | StrOutputParser()
    )

    return standalone_question_chain


def get_reader_chain(llm, reader_template, retriever):
    # 1. Generate the prompt using prompt template
    reader_prompt = ChatPromptTemplate.from_template(template=reader_template)

    # 2. Use LCEL to create a chain of the LLM with the prompt template
    llm_chain = reader_prompt | llm

    # 3. Pass the retrieved documents as the context and the input query to the LLM Chain created in step 2
    reader = {
        "docs": retriever,
        "question": lambda x: x,
    }

    chat_model = {"context": lambda x: _combine_documents(x["docs"]), "question": itemgetter("question")}

    answer = {
        "answer": chat_model | llm_chain,
        "docs": itemgetter("docs"),
    }

    # 4. Return the reader_chain in LCEL
    return reader, answer


def rag_pipeline(query, retriever, query_llm, llm, conversation_template, reader_template, memory):
    # load the memory to access chat history
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    )
    standalone_question_chain = get_conversation_chain(query_llm, conversation_template)
    reader, answer = get_reader_chain(llm, reader_template, retriever)
    rag_chain = loaded_memory | standalone_question_chain | reader | answer
    inputs = {"question": query}
    result = {}
    answer = ""
    curr_key = None
    # result = rag_chain.invoke(inputs)
    for chunk in rag_chain.stream(inputs):
        for key in chunk:
            if key not in result:
                result[key] = chunk[key]
            else:
                result[key] += chunk[key]
            if key == 'answer':
                if key != curr_key:
                    print(
                        colored(f"\nAnswer: {chunk[key].content}", "blue"),
                        end="",
                        flush=True,
                    )
                else:
                    print(colored(chunk[key].content, "blue"), end="", flush=True)
                answer += chunk[key].content
            curr_key = key
    print()
    # save the current question and answer to memory as chat history
    memory.save_context(inputs, {"answer": answer})


if __name__ == "__main__":
    arxiv_id = "1706.03762"
    docs = read_docs(arxiv_id)

    embeddings = HuggingFaceInstructEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL"),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )
    # embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    # FAISS vector database converts the chunks using the embeddings_model
    db = FAISS.from_documents(docs, embeddings)

    # DEFINING RETRIEVER
    retriever = db.as_retriever()

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

    # run pipeline
    questions = [
        "What is attention mechanism",
        "How does it work in self-attention",
        "How is transformer formed",
        "Why do we use Softmax in this",
    ]

    for query in questions:
        print(colored(f"Query: {query}", "red"))
        rag_pipeline(
            query,
            retriever,
            query_llm,
            llm,
            CONVERSATION_TEMPLATE,
            CITATIONS_TEMPLATE,
            memory,
        )
        print("###########" * 4)
