-- ============================================
-- SCRIPT 1: Empty the akili_vectors database
-- WARNING: This deletes ALL data!
-- ============================================

-- Connect to akili_vectors database first, then run:

-- Drop all tables (if they exist)
DROP TABLE IF EXISTS embeddings CASCADE;
DROP TABLE IF EXISTS collections CASCADE;

-- Drop all views
DROP VIEW IF EXISTS collection_stats CASCADE;

-- Drop all functions
DROP FUNCTION IF EXISTS update_collections_updated_at() CASCADE;

-- Drop all triggers (already dropped by CASCADE, but explicitly)
DROP TRIGGER IF EXISTS trigger_collections_updated_at ON collections;

-- Drop pgvector extension (optional - only if you want to reinstall it)
-- DROP EXTENSION IF EXISTS vector CASCADE;

-- Verify everything is deleted
SELECT 
    schemaname, 
    tablename 
FROM pg_tables 
WHERE schemaname = 'public';

-- Should return empty or only system tables