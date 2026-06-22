import asyncio
import sys
import redis.asyncio
import bcrypt

sys.modules["aioredis"] = redis.asyncio


from core.database import init_db
from core.models import Admin


async def main():
    await init_db()
    
    # Hash password using bcrypt (fastapi-admin uses bcrypt internally)
    password = "admin"
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    hashed_password = hashed.decode('utf-8')
    
    # Check if exists
    exists = await Admin.get_or_none(username="admin")
    if not exists:
        await Admin.create(username="admin", password=hashed_password, is_active=True, is_superuser=True)
        print("Admin user created (admin / admin)")
    else:
        print("Admin user already exists")


if __name__ == "__main__":
    asyncio.run(main())