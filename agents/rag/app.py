""" Refer to https://langchain114.com/docs/expression_language/cookbook/retrieval
"""

import os
from pathlib import Path
from operator import itemgetter
from termcolor import colored
from langchain.memory import ConversationBufferMemory
from langchain.schema import format_document
from langchain.schema.messages import get_buffer_string
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.document_loaders import ArxivLoader, PyPDFLoader

os.environ["OPENAI_API_BASE"] = "LINK"
os.environ["OPENAI_API_KEY"] = "KEY"

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
    docs = ArxivLoader(query=arxiv_query).load_and_split()
    print("Total arxiv pages:", len(docs))
    return docs


def get_conversation_chain(llm, template):
    prompt = PromptTemplate.from_template(template=template)

    llm_chain = prompt | llm

    standalone_question_chain = {
        {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | llm_chain
        | StrOutputParser()
    }

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
    result = rag_chain.invoke(inputs)

    # save the current question and answer to memory as chat history
    memory.save_context(inputs, {"answer": result["answer"].content})
    return result["answer"].content


if __name__ == "__main__":
    arxiv_id = "1706.03762"
    docs = read_docs(arxiv_id)

    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large",
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
    CONVERSATION_TEMPLATE = """
    [INST]
    As a Question answering assistant, generate a new question based on the asked question and the conversational history - History.
    History will be passed as a list.
    The string will be in format:
    'Question: '+ asked question + " Answer: "+ answer
    Your task is to fetch the text after keyword 'Question: ' and before keyword "Answer: " from History and this will be the asked question
    by the user. You need to then generate a new question using the conversational history and below guidelines:
    - Conversational history is ordered from least recent to most recent. Weight most recent history the most.
    - If the asked question is not related to the conversational history, do not consider the history while answering and 
    return the original question.
    - Only generate a single query.
    - If multiple options exist for the query, do not separate them by OR and do not ask the user to select. Just pick any single query.
    - Do not ask clarifying questions, if you are not sure about the new query, just output the original question.

    History : {chat_history}
    Asked question : {question}
    New Question: [Your response]
    [/INST]
    """

    # DEFINING READER Prompt Template
    CITATIONS_TEMPLATE = """
    [INST]
    You are QA assistant. You are given a question and a dictionary.
    You need to generate the answer and cite the sentences used in generating the answer.
    The key from dictionary is the citation and the value from the dictionary is the context to generate the answer for the question. 

    - Try your best to list the citations
    - If you do not find the answer, say politely that you don't know.
    - Do not generate false information.
    - Do not combine multiple sources, list them separately like [src_1][src_2]

    Below is an example:
    question : 'When did Roman Empire fall?'
    dictionary: article4.pdf_Page2: The western empire suffered several Gothic invasions and, in AD 455, was sacked by Vandals. Rome continued to decline after that until AD 476 when the western Roman Empire came to an end.
    Answer: 476 CE [article4.pdf_Page2]

    Use the dictionary: {context} for the Question: {question} to generate the answer.
    Helpful Answer: [Your response]
    [/INST]
    """

    query_llm = ChatOpenAI(model="model_name", temperature=0.0)
    llm = ChatOpenAI(model="model_name", temperature=0.2)

    # run pipeline
    questions = [
        "What is attention mechanism",
        "How does it work in self-attention",
        "How is transformer formed",
        "Why do we use Softmax in this",
    ]

    for query in questions:
        print(colored(f"Query: {query}", "red"))
        print(
            colored(
                f"""Answer: {rag_pipeline(query,
                                          retriever,
                                          query_llm,
                                          llm,
                                          CONVERSATION_TEMPLATE,
                                          CITATIONS_TEMPLATE,
                                          memory)}""",
                "blue",
            )
        )
        print("###########" * 4)
