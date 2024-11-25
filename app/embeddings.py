import os
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import OptimizersConfigDiff


class EmbeddingsManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "vector_db",
    ):
        """
        Initializes the EmbeddingsManager with the specified model and Qdrant settings.
        """
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs={"device": device}
        )
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

    def create_embeddings(self, pdf_path: str) -> str:
        """
        Processes a PDF, creates embeddings, and ensures the vector database is cleared before storing new embeddings.

        Args:
            pdf_path (str): Path to the PDF document.

        Returns:
            str: Success message upon completion.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} does not exist.")

        # Load and preprocess the PDF
        loader = UnstructuredPDFLoader(pdf_path)
        docs = loader.load()
        if not docs:
            raise ValueError("No text could be extracted from the PDF.")

        # Split text into manageable chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = splitter.split_documents(docs)
        if not text_chunks:
            raise ValueError("No text chunks were created.")

        try:
            # Connect to Qdrant
            client = QdrantClient(url=self.qdrant_url)

            # Check if the collection exists
            existing_collections = client.get_collections().collections
            collection_names = [collection.name for collection in existing_collections]
            print(collection_names)
            if self.collection_name in collection_names:
                # Delete the existing collection
                client.delete_collection(self.collection_name)
                print("Collection Deleted")
            # Create a new collection without strict_mode_config
            # client.create_collection(
            #     collection_name=self.collection_name,
            #     vectors_config={"size": 768, "distance": "Cosine"},  # Adjust vector size and distance as needed
            #     optimizers_config=OptimizersConfigDiff(max_optimization_threads=2),  # Optimizer configuration
            # )

            # Store new embeddings
            Qdrant.from_documents(
                text_chunks,
                self.embeddings,
                url=self.qdrant_url,
                collection_name=self.collection_name,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {e}")

        return "✅ Embeddings created and replaced in Qdrant successfully!"
