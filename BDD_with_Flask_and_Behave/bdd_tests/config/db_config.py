# bdd_tests/config/db_config.py

import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine.url import URL
from bdd_tests.config.settings import settings

logger = logging.getLogger(__name__)

# 1. Construct the Database URL from the loaded settings
DATABASE_URL = str(URL.create(
    drivername="oracle+oracledb",
    username=settings.DB_USER,
    password=settings.DB_PASSWORD,
    host=settings.DB_HOST,
    port=settings.DB_PORT,
    database=settings.DB_SERVICE_NAME,
))

# 2. Define reusable engine connection options for pooling
engine_options = {
    "pool_size": 10,
    "max_overflow": 20,
    "pool_timeout": 30,
    "pool_recycle": 1800,
    "pool_pre_ping": True,
}

# 3. Create the single, shared engine and Session factory
try:
    engine = create_engine(DATABASE_URL, **engine_options)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("SQLAlchemy engine and session factory created successfully.")
except Exception as e:
    logger.error(f"Failed to create SQLAlchemy engine: {e}")
    raise
  
