from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import pickle


llm = CTransformers(model = "models\codellama-7b-instruct.ggmlv3.Q4_0.bin",
                    model_type = "llama",
                    max_new_tokens=700,
                    temperature=0.2
                    )

file_path = "fiass_index"
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")





def create_vector_db(pdf):
    loader = PyPDFLoader(pdf)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
    chunked_documents = text_splitter.split_documents(pages)
    
    vectordb = FAISS.from_documents(documents=chunked_documents,
                                 embedding=instructor_embeddings)
    retriever = vectordb.as_retriever(score_threshold = 0.7)
    
    vectordb.save_local(file_path)



def get_answer_chain():
    vectordb = FAISS.load_local(file_path, instructor_embeddings)
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=False,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain



if __name__ == "__main__":
    create_vector_db("sem4_receipt.pdf")
    chain = get_answer_chain()
    while True:
        question = input("Ask a question: ")
        answer = chain(question)
        print(answer)

