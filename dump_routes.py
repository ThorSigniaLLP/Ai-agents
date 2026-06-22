import sys
import redis.asyncio as aioredis
sys.modules['aioredis'] = aioredis

from fastapi_admin.app import app
for r in app.routes:
    if hasattr(r, 'path'):
        print("Route:", r.path)
    else:
        print("Other:", r)
