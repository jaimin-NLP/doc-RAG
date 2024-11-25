from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_ollama import ChatOllama
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class ChatbotManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        # llm_model: str = "llama3.2:3b",
        llm_temperature: float = 0.7,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "vector_db",
    ):
        """
        Initializes the ChatbotManager with embedding models, LLM, and vector store.
        """
        # Embeddings for vector retrieval
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs={"device": device}
        )

        # Language Model (LLM)
        # self.llm = ChatOllama(model="llama3.2:3b",temperature=llm_temperature,server_url="http://localhost:3000",)
        self.llm = Ollama(model="llama3.2", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
        # Prompt Template
        self.prompt = PromptTemplate(
            template=(
                "Use the following context to answer the user's question.\n\n"
                "Context: {context}\n\n"
                "Question: {question}\n\n"
                "Helpful answer:"
            ),
            input_variables=["context", "question"],
        )

        # Qdrant vector store
        self.qdrant = Qdrant(
            client=QdrantClient(url=qdrant_url),
            embeddings=self.embeddings,
            collection_name=collection_name,
        )

        # Retriever for fetching relevant text chunks
        self.retriever = self.qdrant.as_retriever(search_kwargs={"k": 2})

        # RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
        )

    def get_response(self, query: str):
        """
        Generates a response to the user's query using the RetrievalQA chain
        and streams the response incrementally.
        
        Args:
            query (str): User's query.

        Yields:
            str: Chatbot's response chunks.
        """
        try:
            response = self.qa_chain.run(query)  # Full response generation
            for chunk in response.split(". "):  # Simulate streaming by splitting sentences
                yield chunk + ". "
        except Exception as e:
            yield f"⚠️ Error generating response: {e}"
