
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE faq_embeddings (
  id UUID PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  content TEXT NOT NULL,
  embedding VECTOR(1536),
  created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX faq_embeddings_idx
ON faq_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE TABLE analytics_logs (
  id SERIAL PRIMARY KEY,
  tenant_id TEXT,
  query TEXT,
  route TEXT,
  latency_ms INT,
  created_at TIMESTAMP DEFAULT now()
);
