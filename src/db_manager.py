import psycopg2
from psycopg2.extras import RealDictCursor, Json
import psycopg2.pool
import pgvector.psycopg2
import logging
import uuid
import json
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DBManager:
    def __init__(self, dbname="test_db", user="root", password="lavita!978", host="localhost", port="5432"):
        self.conn_params = {
            "dbname": dbname,
            "user": user,
            "password": password,
            "host": host,
            "port": port
        }
        # Initialize Connection Pool
        self.pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1, 
            maxconn=20, 
            **self.conn_params
        )
        self._init_db()

    @contextmanager
    def get_conn(self):
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)

    def _init_db(self):
        with self.get_conn() as conn:
            try:
                with conn.cursor() as cur:
                    # Enable pgvector
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    
                    # Create documents table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS documents (
                            id UUID PRIMARY KEY,
                            title TEXT,
                            category TEXT NOT NULL,
                            level TEXT,
                            metadata JSONB,
                            content TEXT,
                            summary_uuids JSONB DEFAULT '[]'::jsonb,
                            source_uuids JSONB DEFAULT '[]'::jsonb,
                            embedding vector(384),
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    
                    # Migration: Add level column if not exists and migrate data
                    try:
                        cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS level TEXT;")
                        cur.execute("""
                            UPDATE documents 
                            SET level = category, category = 'General' 
                            WHERE level IS NULL AND category IN ('L0', 'L1', 'L2', 'L3');
                        """)
                        cur.execute("UPDATE documents SET level = 'L0' WHERE level IS NULL;")
                    except Exception as e:
                        logger.warning(f"Migration error (level): {e}")

                    # Migration: Add title column
                    try:
                        cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS title TEXT;")
                        cur.execute("""
                            UPDATE documents 
                            SET title = SUBSTRING(content FROM 1 FOR 20)
                            WHERE title IS NULL;
                        """)
                    except Exception as e:
                        logger.warning(f"Migration error (title): {e}")

                    # Migration: Add source_uuids
                    try:
                        cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_uuids JSONB DEFAULT '[]'::jsonb;")
                    except Exception as e:
                        logger.warning(f"Migration error (source_uuids): {e}")

                    # Create processing tasks table (QUEUE)
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS processing_tasks (
                            doc_id UUID PRIMARY KEY,
                            status TEXT DEFAULT 'created', -- created, queued, processing, done, failed
                            config JSONB DEFAULT '{}'::jsonb, -- prompts, models
                            results JSONB DEFAULT '{}'::jsonb, -- partial or full results
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                    """)

                    # Indexes
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_category ON documents(category);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_level ON documents(level);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_metadata ON documents USING gin(metadata);")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON processing_tasks(status);")
                    
                    conn.commit()
            except Exception as e:
                logger.error(f"Error initializing DB: {e}")
                conn.rollback()

    def upsert_document(self, doc_id, category, level, meta, content, embedding=None, title=None):
        with self.get_conn() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO documents (id, title, category, level, metadata, content, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            title = EXCLUDED.title,
                            category = EXCLUDED.category,
                            level = EXCLUDED.level,
                            metadata = EXCLUDED.metadata,
                            content = EXCLUDED.content,
                            embedding = COALESCE(EXCLUDED.embedding, documents.embedding);
                    """, (doc_id, title, category, level, Json(meta), content, embedding))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Error upserting document: {e}")
                conn.rollback()
                return False

    def link_documents(self, source_id, summary_id):
        with self.get_conn() as conn:
            try:
                with conn.cursor() as cur:
                    # Add summary_id to source's summary_uuids
                    cur.execute("""
                        UPDATE documents 
                        SET summary_uuids = (
                            SELECT jsonb_agg(DISTINCT value)
                            FROM (
                                SELECT jsonb_array_elements(COALESCE(summary_uuids, '[]'::jsonb)) AS value 
                                FROM documents WHERE id = %s
                                UNION ALL
                                SELECT jsonb_build_array(%s::text)->0
                            ) s
                        )
                        WHERE id = %s;
                    """, (source_id, str(summary_id), source_id))
                    
                    # Add source_id to summary's source_uuids
                    cur.execute("""
                        UPDATE documents 
                        SET source_uuids = (
                            SELECT jsonb_agg(DISTINCT value)
                            FROM (
                                SELECT jsonb_array_elements(COALESCE(source_uuids, '[]'::jsonb)) AS value 
                                FROM documents WHERE id = %s
                                UNION ALL
                                SELECT jsonb_build_array(%s::text)->0
                            ) s
                        )
                        WHERE id = %s;
                    """, (str(summary_id), source_id, str(summary_id)))
                    
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Error linking documents: {e}")
                conn.rollback()
                return False

    def remove_summary_link(self, source_id, summary_id):
        with self.get_conn() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE documents
                        SET summary_uuids = (
                            SELECT COALESCE(jsonb_agg(value), '[]'::jsonb)
                            FROM jsonb_array_elements(summary_uuids) AS value
                            WHERE value #>> '{}' != %s
                        )
                        WHERE id = %s;
                    """, (str(summary_id), source_id))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Error removing summary link: {e}")
                conn.rollback()
                return False

    def clear_summary_links(self, source_id):
        with self.get_conn() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("UPDATE documents SET summary_uuids = '[]'::jsonb WHERE id = %s", (source_id,))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Error clearing summary links: {e}")
                conn.rollback()
                return False

    def add_summary_link(self, parent_id, summary_id):
        return self.link_documents(parent_id, summary_id)

    def delete_document(self, doc_id):
        with self.get_conn() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Error deleting document: {e}")
                conn.rollback()
                return False

    def get_document(self, doc_id):
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
                return cur.fetchone()

    def search_documents(self, query_text=None, category=None, level=None, doc_id=None, metadata_filters=None):
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                sql = "SELECT * FROM documents WHERE 1=1"
                params = []
                
                if doc_id:
                    sql += " AND id = %s"
                    params.append(doc_id)
                if category:
                    sql += " AND category = %s"
                    params.append(category)
                if level:
                    sql += " AND level = %s"
                    params.append(level)
                if query_text:
                    sql += " AND content ILIKE %s"
                    params.append(f"%{query_text}%")
                if metadata_filters:
                    for k, v in metadata_filters.items():
                        sql += " AND metadata->>%s = %s"
                        params.extend([k, str(v)])
                
                sql += " ORDER BY created_at DESC"
                cur.execute(sql, params)
                return cur.fetchall()

    def vector_search(self, embedding, limit=5, category=None, level=None):
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                sql = "SELECT *, 1 - (embedding <=> %s) AS cosine_similarity FROM documents"
                params = [embedding]
                
                where_clauses = []
                if category:
                    where_clauses.append("category = %s")
                    params.append(category)
                if level:
                    where_clauses.append("level = %s")
                    params.append(level)
                
                if where_clauses:
                    sql += " WHERE " + " AND ".join(where_clauses)
                
                sql += " ORDER BY embedding <=> %s LIMIT %s"
                params.extend([embedding, limit])
                
                cur.execute(sql, params)
                return cur.fetchall()

    def enqueue_task(self, doc_id, config=None):
        with self.get_conn() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO processing_tasks (doc_id, status, config)
                        VALUES (%s, 'created', %s)
                        ON CONFLICT (doc_id) DO UPDATE SET
                            status = 'created',
                            config = EXCLUDED.config,
                            updated_at = CURRENT_TIMESTAMP;
                    """, (doc_id, Json(config or {})))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Error enqueueing task: {e}")
                conn.rollback()
                return False

    def update_task(self, doc_id, status=None, results=None, config=None):
        with self.get_conn() as conn:
            try:
                with conn.cursor() as cur:
                    sql = "UPDATE processing_tasks SET updated_at = CURRENT_TIMESTAMP"
                    params = []
                    
                    if status:
                        sql += ", status = %s"
                        params.append(status)
                    if results:
                        sql += ", results = %s"
                        params.append(Json(results))
                    if config:
                        sql += ", config = %s"
                        params.append(Json(config))
                    
                    sql += " WHERE doc_id = %s"
                    params.append(doc_id)
                    
                    cur.execute(sql, params)
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Error updating task: {e}")
                conn.rollback()
                return False

    def get_tasks_by_status(self, status):
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM processing_tasks WHERE status = %s ORDER BY created_at ASC", (status,))
                return cur.fetchall()

    def get_task(self, doc_id):
        with self.get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM processing_tasks WHERE doc_id = %s", (doc_id,))
                return cur.fetchone()

    def delete_task(self, doc_id):
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM processing_tasks WHERE doc_id = %s", (doc_id,))
            conn.commit()
