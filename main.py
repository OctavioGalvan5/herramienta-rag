"""
n8n RAG Tool - API Server
=========================
API Flask que expone el RAG con FlashRank + Multi-Query
para ser consumido por n8n como herramienta.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from config.settings import Config
from rag.retriever import get_advanced_retriever, get_simple_retriever, debug_print
from rag.vectorstore import vectorstore_manager
import time
import os

app = Flask(__name__)
CORS(app)  # Permitir CORS para n8n


@app.route('/api/rag/query', methods=['POST'])
def query_rag():
    """
    Endpoint principal para consultas RAG desde n8n.
    
    Input JSON:
    {
        "query": "La pregunta del usuario",
        "top_k": 5,                   # Opcional
        "mode": "full"                # Opcional: "full", "simple"
    }
    
    Output JSON:
    {
        "context": "Texto formateado para el LLM",
        "sources": [...],
        "documents_count": 5,
        "elapsed_ms": 1234
    }
    """
    start_time = time.time()
    
    # Parse input
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "Se requiere el campo 'query'"}), 400
    
    query = data['query']
    top_k = data.get('top_k', 5)
    mode = data.get('mode', 'full')
    
    debug_print("API", f"Query: '{query[:80]}...'")
    debug_print("API", f"top_k={top_k}, mode={mode}")
    
    try:
        # Seleccionar retriever según modo
        if mode == 'simple':
            retriever = get_simple_retriever(k=top_k)
        else:
            retriever = get_advanced_retriever(
                top_k=top_k,
                use_rerank=True, 
                use_multi_query=True
            )
        
        # Ejecutar retrieval
        docs = retriever.invoke(query)
        
        # Formatear contexto para el LLM
        context_parts = []
        sources = []
        
        for i, doc in enumerate(docs):
            # Info de fuente si existe
            source_info = ""
            if doc.metadata.get('file_title'):
                source_info = f"[{doc.metadata['file_title']}]"
            elif doc.metadata.get('file_id'):
                source_info = f"[Doc: {doc.metadata['file_id'][:8]}]"
            
            context_parts.append(f"--- Documento {i+1} {source_info} ---\n{doc.page_content}")
            
            sources.append({
                "content": doc.page_content[:500],
                "metadata": doc.metadata
            })
        
        context = "\n\n".join(context_parts)
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        debug_print("API", f"✅ Respuesta lista en {elapsed_ms}ms, {len(docs)} documentos")
        
        return jsonify({
            "context": context,
            "sources": sources,
            "query": query,
            "documents_count": len(docs),
            "elapsed_ms": elapsed_ms
        }), 200
        
    except Exception as e:
        debug_print("API", f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "context": "",
            "sources": []
        }), 500


@app.route('/api/rag/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Probar conexión a la DB
        connected = vectorstore_manager.test_connection()
        status = "ok" if connected else "database_error"
    except Exception as e:
        status = "error"
        return jsonify({
            "status": status,
            "error": str(e)
        }), 500
    
    return jsonify({
        "status": status,
        "service": "n8n-rag-tool",
        "database": {
            "host": Config.POSTGRES_HOST,
            "table": Config.PGVECTOR_TABLE
        }
    }), 200


@app.route('/', methods=['GET'])
def index():
    """Root endpoint with info."""
    return jsonify({
        "service": "n8n RAG Tool",
        "version": "1.0.0",
        "endpoints": {
            "/api/rag/query": "POST - Consulta RAG con FlashRank + Multi-Query",
            "/api/rag/health": "GET - Health check"
        }
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 n8n RAG Tool - Iniciando...")
    print(f"📦 PostgreSQL: {Config.POSTGRES_HOST}/{Config.POSTGRES_DATABASE}")
    print(f"📋 Tabla: {Config.PGVECTOR_TABLE}")
    print(f"🌐 Server: http://{Config.HOST}:{Config.PORT}")
    print("=" * 60)
    
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
