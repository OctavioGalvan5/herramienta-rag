"""
PGVector VectorStore Manager
=============================
Conecta al mismo PostgreSQL/PGVector que usa n8n.
Lee documentos de la tabla existente sin modificarla.
"""

from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from config.settings import Config
from typing import List, Optional


class PGVectorManager:
    """
    Manager para conectar al PGVector existente de n8n.
    SOLO LEE documentos, no modifica la tabla.
    """
    
    def __init__(self):
        self._vectorstore = None
        self._embeddings = None
    
    @property
    def embeddings(self):
        """Lazy load embeddings."""
        if self._embeddings is None:
            # Usar el mismo modelo de embeddings que n8n (text-embedding-3-small)
            self._embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=Config.OPENAI_API_KEY
            )
        return self._embeddings
    
    @property
    def vectorstore(self):
        """Lazy load vectorstore connection."""
        if self._vectorstore is None:
            connection_string = Config.get_postgres_uri()
            
            print(f"🔗 Conectando a PGVector: {Config.POSTGRES_HOST}/{Config.POSTGRES_DATABASE}")
            print(f"📋 Tabla: {Config.PGVECTOR_TABLE}")
            
            self._vectorstore = PGVector(
                connection=connection_string,
                embeddings=self.embeddings,
                collection_name=Config.PGVECTOR_TABLE,
                use_jsonb=True,  # n8n usa JSONB para metadata
            )
            
            print("✅ Conexión a PGVector establecida")
        
        return self._vectorstore
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter_dict: Optional[dict] = None
    ) -> List[Document]:
        """
        Búsqueda semántica en la base de datos.
        
        Args:
            query: Texto de búsqueda
            k: Número de resultados
            filter_dict: Filtros de metadata opcionales
        """
        print(f"🔍 Buscando: '{query[:50]}...' (k={k})")
        
        if filter_dict:
            results = self.vectorstore.similarity_search(
                query, 
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vectorstore.similarity_search(query, k=k)
        
        print(f"📦 Encontrados: {len(results)} documentos")
        return results
    
    def get_retriever(self, k: int = 5, filter_dict: Optional[dict] = None):
        """
        Obtiene un retriever de LangChain.
        
        Args:
            k: Número de documentos a retornar
            filter_dict: Filtros de metadata opcionales
        """
        search_kwargs = {"k": k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict
            
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def test_connection(self) -> bool:
        """Prueba la conexión a la base de datos."""
        try:
            # Intentar una búsqueda simple
            results = self.similarity_search("test", k=1)
            print(f"✅ Conexión exitosa. Documentos en DB: disponibles")
            return True
        except Exception as e:
            print(f"❌ Error de conexión: {str(e)}")
            return False


# Singleton instance
vectorstore_manager = PGVectorManager()
