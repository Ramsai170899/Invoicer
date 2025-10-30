```
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA

def create_csv_qa_chain(csv_path, model_name="llama3"):
    """
    Create a RetrievalQA chain from a local CSV file using Ollama LLM and LangChain.
    """
    print(f"\n📂 Loading CSV file: {csv_path}")
    loader = CSVLoader(file_path=csv_path, encoding="utf-8")
    data = loader.load()

    print("🧩 Splitting text into chunks for embedding...")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(data)

    print("🧠 Creating vector store using Ollama embeddings...")
    embeddings = OllamaEmbeddings(model=model_name)
    vectorstore = FAISS.from_documents(docs, embeddings)

    print("🔗 Building retrieval chain...")
    llm = OllamaLLM(model=model_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    print("✅ CSV data loaded and QA chain ready!\n")
    return qa_chain


def chat_with_csv():
    print("\n\t📊 Welcome to CSV Chatbot (powered by Ollama + LangChain)")
    print("\tType 'exit' anytime to quit.\n")

    csv_path = input("Enter the path to your local CSV file: ").strip()
    qa_chain = create_csv_qa_chain(csv_path)

    while True:
        query = input("\n🔎 You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting chat. Goodbye!")
            break

        result = qa_chain.invoke({"query": query})
        print("\n🤖 Bot:", result["result"])

        # Optional: show sources
        sources = [doc.metadata.get("source") for doc in result["source_documents"]]
        print("📁 Sources:", list(set(sources)))


if __name__ == "__main__":
    chat_with_csv()

```
