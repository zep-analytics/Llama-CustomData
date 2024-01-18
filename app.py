from flask import Flask, render_template, request
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers

app = Flask(__name__)

# Load PDF documents from the 'data/' directory
loader = DirectoryLoader('data1/', glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# Load embedding model and create a FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
vector_store = FAISS.from_documents(text_chunks, embeddings)

# Load LLM model for question answering
llm = CTransformers(model="models\llama-2-7b-chat.ggmlv3.q8_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 128, 'temperature': 0.01})

# Define the prompt template for question answering
template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say you don't know; don't try to make up an answer.

Context:{context}
Question:{question}

Only return the helpful answer below and nothing else
Helpful answer
"""

qa_prompt = PromptTemplate(template=template, input_variables=['context', 'question'])

# Create a RetrievalQA chain
chain = RetrievalQA.from_chain_type(llm=llm,
                                   chain_type='stuff',
                                   retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
                                   return_source_documents=True,
                                   chain_type_kwargs={'prompt': qa_prompt})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    # Save the uploaded PDF file
    file.save('uploaded_file.pdf')

    return render_template('index.html', success='File uploaded successfully')


@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form.get('question')
    result = chain({'query': question})
    answer = result['result']
    return render_template('index.html', answer=answer)


if __name__ == '__main__':
    app.run(debug=True)
