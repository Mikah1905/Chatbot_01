from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

class RAGModule:
    def __init__(self, pdf_path):
        # Load PDF and extract text
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()

        # Split document into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.docs = splitter.split_documents(documents)

        # Embed the chunks
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = FAISS.from_documents(self.docs, embeddings)

    def retrieve_context(self, query, k=3):
        results = self.db.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in results])
