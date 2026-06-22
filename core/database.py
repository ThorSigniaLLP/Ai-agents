import logging
from tortoise import Tortoise
from core.config import get_settings

logger = logging.getLogger(__name__)

TORTOISE_ORM = {
    "connections": {"default": ""},
    "apps": {
        "models": {
            "models": ["core.models"],
            "default_connection": "default",
        },
    },
    "use_tz": False,
}

async def init_db():
    settings = get_settings()
    # ensure asyncpg connection format (postgresql:// to postgres://)
    db_url = settings.database_url.replace("postgresql://", "postgres://")
    TORTOISE_ORM["connections"]["default"] = db_url
    
    try:
        await Tortoise.init(config=TORTOISE_ORM, _enable_global_fallback=True)
        await Tortoise.generate_schemas(safe=True)
        logger.info("[Database] Connected to TortoiseORM.")
    except Exception as e:
        logger.error(f"[Database] Failed to connect: {e}")

async def close_db():
    await Tortoise.close_connections()
