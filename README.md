# n8n RAG Tool 🐍

Herramienta RAG personalizada para n8n que se conecta a la misma base de datos PostgreSQL/PGVector,
pero usa **FlashRank** y **Multi-Query** para mejores resultados.

## Ventajas sobre el RAG nativo de n8n

| Característica | n8n Nativo | Este Tool |
|---------------|------------|-----------|
| Reranking | Cohere API (lento, $) | FlashRank local (gratis, ~100x más rápido) |
| Multi-Query | ❌ No | ✅ Sí |
| Deduplicación | ❌ No | ✅ Sí |
| Base de datos | PGVector | PGVector (misma!) |

## Setup

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Configurar variables de entorno
```bash
cp .env.example .env
# Editar .env con las credenciales de PostgreSQL
```

### 3. Ejecutar
```bash
python main.py
```

## Uso desde n8n

### Endpoint principal
```
POST /api/rag/query
```

### Ejemplo de llamada HTTP en n8n:
```json
{
  "query": "¿Cómo tramitar el carnet de afiliado?",
  "top_k": 5,
  "mode": "full"
}
```

### Respuesta:
```json
{
  "context": "--- Documento 1 ---\n...",
  "sources": [...],
  "documents_count": 5,
  "elapsed_ms": 450
}
```

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/api/rag/query` | Consulta RAG |
| GET | `/api/rag/health` | Health check |
| GET | `/` | Info del servicio |
