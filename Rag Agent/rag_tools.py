import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------
# 1. Load PDF
# -------------------------

pdf_path = "Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()

    print(f"PDF has been loaded and has {len(pages)} pages")

except Exception as e:
    print(f"Error loading PDF: {e}")
    raise


# -------------------------
# 2. Split PDF into Chunks
# -------------------------

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

pages_split = text_splitter.split_documents(pages)

print(f"Created {len(pages_split)} document chunks")


# -------------------------
# 3. Create Embeddings
# -------------------------

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")


# -------------------------
# 4. Create ChromaDB
# -------------------------

persist_directory = r"C:\MY FOLDERS\Project\Agents\Rag Agent"

collection_name = "stock_market"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    print("Created ChromaDB vector store!")

except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")

    raise


# -------------------------
# 5. Create Retriever
# -------------------------

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


# -------------------------
# 6. Retriever Tool
# -------------------------


@tool
def retriever_tool(query: str) -> str:
    """
    Search the Stock Market Performance 2024
    document and return relevant information.
    """

    docs = retriever.invoke(query)

    if not docs:
        return (
            "I found no relevant information in "
            "the Stock Market Performance 2024 document."
        )

    results = []

    for i, doc in enumerate(docs):
        results.append(f"Document {i + 1}:\n{doc.page_content}")

    return "\n\n".join(results)


# -------------------------
# 7. Tools List
# -------------------------

tools = [retriever_tool]
