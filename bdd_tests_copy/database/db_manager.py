import oracledb
from typing import Optional, Dict, List, Any
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool

from ..config.settings import Settings
from ..utils.logger_config import get_logger


class DatabaseManager:
    """
    Manages Oracle database connections using both oracledb and SQLAlchemy.
    Handles the "service is not a registered listener" error properly.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger(__name__)
        self._engine = None
        self._session_factory = None
        self._oracledb_pool = None
        self._initialize()
    
    def _initialize(self):
        """Initialize both SQLAlchemy and oracledb connections."""
        try:
            # Initialize SQLAlchemy
            self._init_sqlalchemy()
            
            # Initialize oracledb pool
            self._init_oracledb_pool()
            
            self.logger.info("Database manager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize database manager: {e}")
            raise
    
    def _init_sqlalchemy(self):
        """Initialize SQLAlchemy engine with proper Oracle configuration."""
        try:
            # Use the connection string from settings
            connection_string = self.settings.DB_CONNECTION_STRING
            
            self.logger.info(f"Initializing SQLAlchemy with host: {self.settings.DB_HOST}")
            
            # Create engine with proper Oracle settings
            self._engine = create_engine(
                connection_string,
                poolclass=NullPool,  # Disable connection pooling to avoid conflicts
                echo=self.settings.ENABLE_DB_QUERY_LOGGING,
                connect_args={
                    "encoding": "UTF-8",
                    "nencoding": "UTF-8",
                    "mode": oracledb.SYSDBA if self.settings.DB_USER.upper() == 'SYS' else oracledb.DEFAULT_AUTH,
                    "events": True,
                    "threaded": True
                }
            )
            
            # Test connection
            with self._engine.connect() as conn:
                result = conn.execute(text("SELECT 1 FROM DUAL"))
                result.fetchone()
            
            # Create session factory
            self._session_factory = sessionmaker(bind=self._engine, autocommit=False, autoflush=False)
            
            self.logger.info("SQLAlchemy engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize SQLAlchemy: {e}")
            raise
    
    def _init_oracledb_pool(self):
        """Initialize oracledb connection pool for direct queries."""
        try:
            # Create DSN using makedsn to ensure proper format
            dsn = oracledb.makedsn(
                host=self.settings.DB_HOST,
                port=self.settings.DB_PORT,
                service_name=self.settings.DB_SERVICE_NAME
            )
            
            self.logger.info(f"Creating oracledb pool with DSN: {dsn}")
            
            # Create connection pool
            self._oracledb_pool = oracledb.create_pool(
                user=self.settings.DB_USER,
                password=self.settings.DB_PRD,
                dsn=dsn,
                min=2,
                max=self.settings.DB_POOL_SIZE,
                increment=1,
                encoding="UTF-8",
                nencoding="UTF-8"
            )
            
            # Test pool
            with self._oracledb_pool.acquire() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1 FROM DUAL")
                    cursor.fetchone()
            
            self.logger.info("OracleDB pool initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize oracledb pool: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Session:
        """Get SQLAlchemy session with context manager."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    @contextmanager
    def get_connection(self):
        """Get oracledb connection from pool."""
        conn = self._oracledb_pool.acquire()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Database connection error: {e}")
            raise
        finally:
            self._oracledb_pool.release(conn)
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Execute query and return results as list of dictionaries."""
        with self.get_session() as session:
            result = session.execute(text(query), params or {})
            
            # Convert to list of dictionaries
            if result.returns_rows:
                columns = result.keys()
                return [dict(zip(columns, row)) for row in result.fetchall()]
            return []
    
    def close(self):
        """Close all database connections."""
        if self._engine:
            self._engine.dispose()
            self.logger.info("SQLAlchemy engine disposed")
        
        if self._oracledb_pool:
            self._oracledb_pool.close()
            self.logger.info("OracleDB pool closed")