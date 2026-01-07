"""
PGVector VectorStore Manager para n8n
=====================================
Consulta directamente la tabla documents_pg que usa n8n.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from config.settings import Config
from typing import List, Optional
import json


class PGVectorManager:
    """
    Manager para conectar al PGVector de n8n.
    Consulta directamente la tabla documents_pg.
    """
    
    def __init__(self):
        self._embeddings = None
        self._connection = None
    
    @property
    def embeddings(self):
        """Lazy load embeddings."""
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=Config.OPENAI_API_KEY
            )
        return self._embeddings
    
    def get_connection(self):
        """Get database connection."""
        try:
            conn = psycopg2.connect(
                host=Config.POSTGRES_HOST,
                port=Config.POSTGRES_PORT,
                user=Config.POSTGRES_USER,
                password=Config.POSTGRES_PASSWORD,
                database=Config.POSTGRES_DATABASE
            )
            return conn
        except Exception as e:
            print(f"❌ Error conectando a PostgreSQL: {e}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter_dict: Optional[dict] = None
    ) -> List[Document]:
        """
        Búsqueda semántica en la tabla documents_pg de n8n.
        
        Args:
            query: Texto de búsqueda
            k: Número de resultados
            filter_dict: Filtros de metadata opcionales (no implementado aún)
        """
        print(f"🔍 Buscando: '{query[:50]}...' (k={k})")
        
        # Generar embedding de la query
        query_embedding = self.embeddings.embed_query(query)
        
        # Formatear embedding para PostgreSQL
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
        
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Consultar tabla documents_pg de n8n
                # La tabla tiene: id, content, metadata, embedding
                sql = f"""
                    SELECT 
                        id,
                        text as content,
                        metadata,
                        1 - (embedding <=> %s::vector) as similarity
                    FROM {Config.PGVECTOR_TABLE}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """
                
                cur.execute(sql, (embedding_str, embedding_str, k))
                rows = cur.fetchall()
                
                print(f"📦 Encontrados: {len(rows)} documentos")
                
                # Convertir a Documents de LangChain
                documents = []
                for row in rows:
                    metadata = row.get('metadata', {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            metadata = {}
                    
                    # Agregar similarity score a metadata
                    metadata['similarity'] = float(row.get('similarity', 0))
                    
                    doc = Document(
                        page_content=row.get('content', ''),
                        metadata=metadata
                    )
                    documents.append(doc)
                
                return documents
                
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
            # Intentar ver la estructura de la tabla
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = '{Config.PGVECTOR_TABLE}'
                    """)
                    columns = cur.fetchall()
                    print(f"📋 Estructura de tabla {Config.PGVECTOR_TABLE}:")
                    for col in columns:
                        print(f"   - {col['column_name']}: {col['data_type']}")
            except Exception as e2:
                print(f"   Error obteniendo estructura: {e2}")
            raise
        finally:
            conn.close()
    
    def get_retriever(self, k: int = 5, filter_dict: Optional[dict] = None):
        """
        Obtiene un retriever simple.
        
        Args:
            k: Número de documentos a retornar
            filter_dict: Filtros de metadata opcionales
        """
        # Retornar un wrapper que permite usar similarity_search
        return SimpleRetriever(self, k=k, filter_dict=filter_dict)
    
    def test_connection(self) -> bool:
        """Prueba la conexión a la base de datos."""
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {Config.PGVECTOR_TABLE}")
                count = cur.fetchone()[0]
                print(f"✅ Conexión exitosa. Documentos en {Config.PGVECTOR_TABLE}: {count}")
            conn.close()
            return True
        except Exception as e:
            print(f"❌ Error de conexión: {str(e)}")
            return False


class SimpleRetriever:
    """Wrapper simple para compatibilidad con LangChain."""
    
    def __init__(self, manager: PGVectorManager, k: int = 5, filter_dict: Optional[dict] = None):
        self.manager = manager
        self.k = k
        self.filter_dict = filter_dict
    
    def invoke(self, query: str) -> List[Document]:
        """Ejecutar búsqueda."""
        return self.manager.similarity_search(query, k=self.k, filter_dict=self.filter_dict)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Método legacy de LangChain."""
        return self.invoke(query)


# Singleton instance
vectorstore_manager = PGVectorManager()
