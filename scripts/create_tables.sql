-- ============================================
-- SCRIPT 2: Create Vector Database Schema
-- Run this after emptying the database
-- ============================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create collections table (metadata tracking)
CREATE TABLE collections (
    id SERIAL PRIMARY KEY,
    collection_name VARCHAR(255) UNIQUE NOT NULL,
    business_id VARCHAR(255) NOT NULL,
    chatbot_id VARCHAR(255) NOT NULL,
    
    -- Document counts (cached for performance)
    document_count INTEGER DEFAULT 0,
    faq_document_count INTEGER DEFAULT 0,
    database_document_count INTEGER DEFAULT 0,
    
    -- Metadata
    vector_dimension INTEGER DEFAULT 768,
    avg_document_size INTEGER DEFAULT 0,
    
    -- Audit
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(business_id, chatbot_id)
);

-- 3. Create embeddings table (main vector storage)
CREATE TABLE embeddings (
    id BIGSERIAL PRIMARY KEY,
    doc_id VARCHAR(500) UNIQUE NOT NULL,
    business_id VARCHAR(255) NOT NULL,
    chatbot_id VARCHAR(255) NOT NULL,
    
    -- Vector data (768 dimensions for all-mpnet-base-v2)
    embedding vector(768) NOT NULL,
    
    -- Content
    text TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Create indexes for collections
CREATE INDEX idx_collections_business ON collections(business_id);
CREATE INDEX idx_collections_chatbot ON collections(business_id, chatbot_id);
CREATE INDEX idx_collections_updated ON collections(updated_at DESC);

-- 5. Create indexes for embeddings (CRITICAL for performance)
CREATE INDEX idx_embeddings_doc_id ON embeddings(doc_id);
CREATE INDEX idx_embeddings_business_chatbot ON embeddings(business_id, chatbot_id);
CREATE INDEX idx_embeddings_business ON embeddings(business_id);

-- Vector similarity search index (HNSW for fast cosine similarity)
CREATE INDEX idx_embeddings_vector ON embeddings 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Metadata indexes for filtering
CREATE INDEX idx_embeddings_source_type ON embeddings 
    USING gin ((metadata->'source_type'));
    
CREATE INDEX idx_embeddings_faq_source ON embeddings 
    USING gin ((metadata->'faq_source_id'));

CREATE INDEX idx_embeddings_table ON embeddings 
    USING gin ((metadata->'table'));

-- 6. Create function to auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 7. Create triggers for auto-updating timestamps
CREATE TRIGGER trigger_embeddings_updated_at
    BEFORE UPDATE ON embeddings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_collections_updated_at
    BEFORE UPDATE ON collections
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 8. Create useful views
CREATE OR REPLACE VIEW collection_stats AS
SELECT 
    c.collection_name,
    c.business_id,
    c.chatbot_id,
    c.document_count as tracked_count,
    COUNT(e.doc_id) as actual_count,
    c.faq_document_count,
    c.database_document_count,
    c.avg_document_size,
    c.created_at,
    c.updated_at,
    CASE 
        WHEN c.document_count = COUNT(e.doc_id) THEN 'synced'
        WHEN c.document_count > COUNT(e.doc_id) THEN 'overcounted'
        ELSE 'undercounted'
    END as sync_status
FROM collections c
LEFT JOIN embeddings e ON c.business_id = e.business_id 
    AND c.chatbot_id = e.chatbot_id
GROUP BY c.id, c.collection_name, c.business_id, c.chatbot_id, 
    c.document_count, c.faq_document_count, c.database_document_count,
    c.avg_document_size, c.created_at, c.updated_at
ORDER BY c.updated_at DESC;

-- 9. Verify setup
SELECT 'Tables created:' as status;
SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;

SELECT 'Indexes created:' as status;
SELECT indexname FROM pg_indexes WHERE schemaname = 'public' ORDER BY indexname;

SELECT 'Setup complete!' as status;

-- Test queries (run these to verify everything works)
-- INSERT INTO collections (collection_name, business_id, chatbot_id) 
-- VALUES ('test_collection', 'test_business', 'test_chatbot');

-- SELECT * FROM collections;
-- SELECT * FROM collection_stats;