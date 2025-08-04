import contextlib
from typing import Generator, Optional, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine

from config.settings import Settings
from core.utils.logger import get_logger

class DatabaseManager:
    """Enterprise-grade database connection manager with connection pooling."""

    def __init__(self, config: Settings):
        self.config = config
        self.logger = get_logger(__name__)
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._initialize()

    def _initialize(self):
        """Initialize database engine with connection pooling."""
        try:
            # Build connection URL
            db_url = self.config.DB_CONNECTION_STRING
            
            if not db_url:
                raise ValueError("Database connection string not configured")
            
            # Create engine with optimized settings
            self._engine = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=self.config.DB_POOL_SIZE,
                max_overflow=self.config.DB_MAX_OVERFLOW,
                pool_timeout=30,
                pool_recycle=1800,
                pool_pre_ping=True,
                echo=self.config.ENABLE_DB_QUERY_LOGGING
            )
            
            # Create session factory
            self._session_factory = sessionmaker(
                bind=self._engine,
                autocommit=False,
                autoflush=False
            )
            
            # Test connection
            with self._engine.connect() as conn:
                result = conn.execute(text("SELECT 1 FROM DUAL"))
                result.fetchone()
            
            self.logger.info(f"Database connection established: {self.config.DB_HOST}:{self.config.DB_PORT}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise

    @contextlib.contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session with automatic cleanup."""
        if not self._session_factory:
            raise RuntimeError("Database not initialized")
        
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database transaction failed: {e}")
            raise
        finally:
            session.close()

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> list:
        """Execute a query and return results."""
        with self.get_session() as session:
            result = session.execute(text(query), params or {})
            return [dict(row) for row in result.mappings()]

    def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1 FROM DUAL"))
            return True
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return False

    def close(self):
        """Close database connections and clean up resources."""
        if self._engine:
            self._engine.dispose()
            self.logger.info("Database connections closed")
