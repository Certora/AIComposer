-- init-db.sql
-- Create separate users for each application
CREATE USER rag_user WITH PASSWORD 'rag_password';
CREATE USER langgraph_store_user WITH PASSWORD 'langgraph_store_password';
CREATE USER langgraph_checkpoint_user WITH PASSWORD 'langgraph_checkpoint_password';

-- Create application-specific databases
CREATE DATABASE rag_db OWNER rag_user;
CREATE DATABASE langgraph_store_db OWNER langgraph_store_user;
CREATE DATABASE langgraph_checkpoint_db OWNER langgraph_checkpoint_user;

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
