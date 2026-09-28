"""
alembic/env.py — Cấu hình Alembic dùng models SQLAlchemy của dự án.
"""
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Import settings và Base để Alembic nhận diện tất cả models
from app.config import get_settings
from app.database import Base

# Import tất cả models để metadata được đăng ký
import app.models  # noqa: F401

settings = get_settings()
config = context.config

# Ghi đè URL từ settings (dùng sync driver psycopg2 cho Alembic)
config.set_main_option("sqlalchemy.url", settings.SYNC_DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Chạy migrations trong 'offline' mode (không cần kết nối DB thực)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Chạy migrations dùng sync engine (psycopg2)."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
