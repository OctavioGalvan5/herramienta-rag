"""
Advanced Retriever con FlashRank + Multi-Query
==============================================
Sistema de recuperación avanzado para n8n.
Adaptado del proyecto original RAG con LangChain.
"""

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List, Optional
import time

# FlashRank para reranking rápido local
try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False
    print("⚠️ FlashRank no instalado. Instalar con: pip install flashrank")

# Multi-Query
try:
    from langchain.retrievers import MultiQueryRetriever
    MULTI_QUERY_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.retrievers import MultiQueryRetriever
        MULTI_QUERY_AVAILABLE = True
    except ImportError:
        MULTI_QUERY_AVAILABLE = False
        print("⚠️ MultiQueryRetriever no disponible")


# ==========================================
# DEBUG UTILITIES
# ==========================================
DEBUG = True

def debug_print(stage: str, message: str, data=None):
    """Print debug information with stage prefix."""
    if DEBUG:
        print(f"\n{'='*60}")
        print(f"🔍 [{stage}] {message}")
        if data is not None:
            if isinstance(data, list):
                print(f"   📦 Count: {len(data)}")
                for i, item in enumerate(data[:3]):
                    if hasattr(item, 'page_content'):
                        content_preview = item.page_content[:100].replace('\n', ' ')
                        print(f"   [{i}] {content_preview}...")
                    else:
                        print(f"   [{i}] {str(item)[:100]}")
                if len(data) > 3:
                    print(f"   ... y {len(data) - 3} más")
            else:
                print(f"   📄 {str(data)[:200]}")
        print(f"{'='*60}\n")


class FlashRankReranker:
    """
    Reranker ultra-rápido usando FlashRank (local, sin API).
    ~100x más rápido que reranking con LLM.
    """
    
    def __init__(self, top_k: int = 5, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        self.top_k = top_k
        self.model_name = model_name
        self._ranker = None
        
    @property
    def ranker(self):
        """Lazy load the ranker model."""
        if self._ranker is None:
            self._ranker = Ranker(model_name=self.model_name)
        return self._ranker
    
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return documents
        
        if not FLASHRANK_AVAILABLE:
            debug_print("RERANK", "⚠️ FlashRank no disponible, retornando orden original")
            return documents[:self.top_k]
        
        debug_print("RERANK", f"FlashRank reranking {len(documents)} documentos...")
        start_time = time.time()
        
        # Preparar passages para FlashRank
        passages = []
        for i, doc in enumerate(documents):
            passages.append({
                "id": i,
                "text": doc.page_content[:2000],
                "meta": doc.metadata
            })
        
        # Crear request de rerank
        rerank_request = RerankRequest(query=query, passages=passages)
        
        # Ejecutar rerank
        results = self.ranker.rerank(rerank_request)
        
        # Mapear de vuelta a documentos
        reranked_docs = []
        for result in results[:self.top_k]:
            original_idx = result["id"]
            reranked_docs.append(documents[original_idx])
        
        elapsed = time.time() - start_time
        debug_print("RERANK", f"✅ FlashRank completado en {elapsed*1000:.0f}ms")
        
        return reranked_docs


class DedupRetriever(BaseRetriever):
    """Elimina documentos duplicados basándose en contenido."""
    
    retriever: BaseRetriever
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Get relevant documents with deduplication."""
        if hasattr(self.retriever, 'invoke'):
            docs = self.retriever.invoke(query)
        else:
            docs = self.retriever.get_relevant_documents(query)
        
        # Deduplicar basándose en contenido
        seen_content = set()
        unique_docs = []
        
        for doc in docs:
            # Usar primeros 200 chars como fingerprint
            fingerprint = doc.page_content[:200].strip()
            if fingerprint not in seen_content:
                seen_content.add(fingerprint)
                unique_docs.append(doc)
        
        if len(docs) != len(unique_docs):
            debug_print("DEDUP", f"Eliminados {len(docs) - len(unique_docs)} duplicados")
        
        return unique_docs


class FlashRankRetriever(BaseRetriever):
    """
    Retriever que aplica FlashRank reranking.
    Compatible con interfaz Runnable de LangChain.
    """
    
    retriever: BaseRetriever
    top_k: int = 5
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Get documents and rerank them with FlashRank."""
        # Obtener documentos del retriever base
        if hasattr(self.retriever, 'invoke'):
            docs = self.retriever.invoke(query)
        else:
            docs = self.retriever.get_relevant_documents(query)
        
        if not docs or not FLASHRANK_AVAILABLE:
            return docs[:self.top_k]
        
        # Reranking con FlashRank
        reranker = FlashRankReranker(top_k=self.top_k)
        return reranker.rerank(query, docs)


def get_advanced_retriever(top_k: int = 5, use_rerank: bool = True, use_multi_query: bool = True):
    """
    Construye el retriever avanzado con todas las optimizaciones:
    1. Base: PGVector (misma DB que n8n)
    2. Multi-Query Retrieval (expansión de consultas)
    3. Deduplicación
    4. FlashRank Re-ranking (rápido, local)
    
    Args:
        top_k: Número de documentos finales a retornar
        use_rerank: Usar FlashRank reranking
        use_multi_query: Usar expansión multi-query
    """
    from rag.vectorstore import vectorstore_manager
    
    debug_print("CHAIN", "Construyendo retriever avanzado...")
    debug_print("CHAIN", f"Opciones: rerank={use_rerank}, multi_query={use_multi_query}, top_k={top_k}")
    
    # Recuperar más docs inicialmente para filtrar después
    # Aumentamos a 50 para dar más opciones al Reranker (calidad similar a Cohere)
    initial_k = 50 if use_rerank else top_k
    
    # 1. Base Retriever (PGVector)
    base_retriever = vectorstore_manager.get_retriever(k=initial_k)
    debug_print("CHAIN", f"✅ Base retriever creado (k={initial_k})")
    
    current_retriever = base_retriever
    
    # 2. Multi-Query Retriever (opcional)
    if use_multi_query and MULTI_QUERY_AVAILABLE:
        # Usar modelo rápido para generar queries alternativas
        helper_llm = ChatOpenAI(temperature=0, model_name="gpt-5.1")
        
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=current_retriever,
            llm=helper_llm
        )
        current_retriever = multi_query_retriever
        debug_print("CHAIN", "✅ Multi-Query retriever agregado")
    
    # 3. Deduplicación
    current_retriever = DedupRetriever(retriever=current_retriever)
    debug_print("CHAIN", "✅ Deduplicación agregada")
    
    # 4. FlashRank reranking
    if use_rerank and FLASHRANK_AVAILABLE:
        current_retriever = FlashRankRetriever(
            retriever=current_retriever, 
            top_k=top_k
        )
        debug_print("CHAIN", f"✅ FlashRank reranker agregado (top_k={top_k})")
    
    debug_print("CHAIN", "✅ Retriever avanzado listo!")
    
    return current_retriever


def get_simple_retriever(k: int = 5):
    """
    Retriever simple sin optimizaciones.
    Útil para debugging o cuando la velocidad es crítica.
    """
    from rag.vectorstore import vectorstore_manager
    
    debug_print("SIMPLE", f"Creando retriever simple (k={k})")
    return vectorstore_manager.get_retriever(k=k)
