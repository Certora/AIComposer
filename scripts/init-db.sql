-- init-db.sql
-- Create separate users for each application
CREATE USER rag_user WITH PASSWORD 'rag_password';
CREATE USER langgraph_store_user WITH PASSWORD 'langgraph_store_password';
CREATE USER langgraph_checkpoint_user WITH PASSWORD 'langgraph_checkpoint_password';
CREATE USER audit_db_user WITH PASSWORD 'audit_db_password';
CREATE USER memory_tool_user WITH PASSWORD 'memory_tool_password';

-- Create application-specific databases
CREATE DATABASE rag_db OWNER rag_user;
CREATE DATABASE langgraph_store_db OWNER langgraph_store_user;
CREATE DATABASE langgraph_checkpoint_db OWNER langgraph_checkpoint_user;
CREATE DATABASE audit_db OWNER audit_db_user;
CREATE DATABASE memory_tool_db OWNER memory_tool_user;

\c rag_db
CREATE EXTENSION IF NOT EXISTS vector;
GRANT ALL PRIVILEGES ON DATABASE rag_db TO rag_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO rag_user;

\c langgraph_store_db
CREATE EXTENSION IF NOT EXISTS vector;
GRANT ALL PRIVILEGES ON DATABASE langgraph_store_db TO langgraph_store_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO langgraph_store_user;

\c langgraph_checkpoint_db
GRANT ALL PRIVILEGES ON DATABASE langgraph_checkpoint_db TO langgraph_checkpoint_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO langgraph_checkpoint_user;

\c audit_db
GRANT ALL PRIVILEGES ON DATABASE audit_db TO audit_db_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO audit_db_user;

\c audit_db
GRANT ALL PRIVILEGES ON DATABASE memory_tool_db TO memory_tool_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO memory_tool_user;

-- Create audit_db schema
CREATE TABLE IF NOT EXISTS file_blobs(
    file_id VARCHAR(64) PRIMARY KEY,
    file_blob BYTEA NOT NULL
);

CREATE TABLE IF NOT EXISTS run_info(
    thread_id TEXT NOT NULL PRIMARY KEY,
    spec_id VARCHAR(64) NOT NULL REFERENCES file_blobs(file_id),
    spec_name TEXT NOT NULL,
    interface_id VARCHAR(64) NOT NULL REFERENCES file_blobs(file_id),
    interface_name TEXT NOT NULL,
    system_id VARCHAR(64) NOT NULL REFERENCES file_blobs(file_id),
    system_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vfs_initial(
    thread_id TEXT NOT NULL REFERENCES run_info(thread_id),
    path TEXT NOT NULL,
    file_id VARCHAR(64) REFERENCES file_blobs(file_id),
    CONSTRAINT vfs_initial_pk PRIMARY KEY(thread_id, path)
);

CREATE INDEX IF NOT EXISTS vfs_init_thread_idx ON vfs_initial(thread_id);

CREATE TABLE IF NOT EXISTS vfs_result(
    thread_id TEXT NOT NULL REFERENCES run_info(thread_id),
    path TEXT NOT NULL,
    file_id VARCHAR(64) REFERENCES file_blobs(file_id),
    CONSTRAINT vfs_result_pk PRIMARY KEY(thread_id, path)
);

CREATE INDEX IF NOT EXISTS vfs_thread_idx on vfs_result(thread_id);

CREATE TABLE IF NOT EXISTS resume_artifact(
    thread_id TEXT NOT NULL PRIMARY KEY REFERENCES run_info(thread_id),
    interface_path TEXT NOT NULL,
    commentary TEXT NOT NULL,
    CONSTRAINT thread_interface_fk FOREIGN KEY (thread_id, interface_path) REFERENCES vfs_result(thread_id, path)
);

CREATE TABLE IF NOT EXISTS prover_results(
    tool_id TEXT NOT NULL,
    rule_name TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    result TEXT NOT NULL CHECK (result in ('VIOLATED', 'ERROR', 'TIMEOUT', 'VERIFIED')),
    analysis TEXT,
    CONSTRAINT prover_results_pk PRIMARY KEY (tool_id, rule_name, thread_id)
);

CREATE TABLE IF NOT EXISTS manual_results(
    tool_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    similarity FLOAT NOT NULL,
    text_body TEXT NOT NULL,
    header_string TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS manual_result_idx ON manual_results (tool_id, thread_id);

CREATE TABLE IF NOT EXISTS summarization(
    thread_id TEXT NOT NULL REFERENCES run_info(thread_id),
    checkpoint_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    CONSTRAINT summarization_pk PRIMARY KEY (thread_id, checkpoint_id)
);

-- Grant permissions to audit_db_user on all tables
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO audit_db_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO audit_db_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL PRIVILEGES ON TABLES TO audit_db_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL PRIVILEGES ON SEQUENCES TO audit_db_user;
