import os
from pathlib import Path
from termcolor import colored
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.document_loaders import ArxivLoader, PyPDFLoader

os.environ["OPENAI_API_BASE"] = "LINK"
os.environ["OPENAI_API_KEY"] = "KEY"

def read_pdf(file_path):
    pdf_search = Path(file_path).glob("*.pdf")
    pdf_files  = [str(file.absolute()) for file in pdf_search]
    print('Total PDF files',len(pdf_files))
    pages = []
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        pages.extend(loader.load_and_split())
    return pages

def read_docs(arxiv_query: str):
    docs = ArxivLoader(query=arxiv_query).load_and_split()
    print('Total arxiv pages:',len(docs))
    return docs

def retrieve_docs(retriever, query, topk):
    return retriever.get_relevant_documents(query, k=topk)

def qa_reader(llm, reader_template, retriever):
    #1. Generate the prompt using prompt template
    reader_prompt = PromptTemplate(template=reader_template, input_variables=["context", "question"])
    
    #2. Use LCEL to create a chain of the LLM with the prompt template
    llm_chain = reader_prompt | llm
    
    #3. Pass the retrieved documents as the context and the input query to the LLM Chain created in step 2
    chat_model = {"context": retriever, "question": RunnablePassthrough()}
    
    #4. Return the rag_chain in LCEL
    rag_chain = (
        chat_model
        | llm_chain
        | StrOutputParser()
    )
    return rag_chain

def rag_pipeline(query, retriever, llm, reader_template):
    rag_chain = qa_reader(llm, reader_template, retriever)
    return rag_chain.invoke(query)

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

    # DEFINING READER LLM
    reader_template = """As a Question answering assitant, generate an answer to the input question using the context provided.
    Follow the below guidelines while answering the question.
    - Use the context to answer the question. Do not answer out of the context available.
    - Be concise and clear in your language.
    - If you do not know the answer just say you - "Sorry, I do not know this!"
    Use the context: {context} for the question: {question} to generate the answer.
    Helpful Answer:"""

    llm = ChatOpenAI(model="model_name")

    # run pipeline
    questions = [
        "What is attention mechanism",
        "What is self-attention",
        "How is transformer formed",
        "Why do we use Softmax in Transformers",
    ]

    for query in questions:
        print(
            colored(
                f"Query: {query}", "red"
            )
        )
        print(
            colored(
                f"Answer: {rag_pipeline(query, retriever, llm, reader_template)}",
                "blue",
            )
        )
        print("###########" * 4)
