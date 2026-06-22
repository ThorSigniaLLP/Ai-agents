import sys
import redis.asyncio as aioredis
sys.modules['aioredis'] = aioredis

from fastapi_admin.app import app

def print_routes(r, prefix=""):
    if hasattr(r, "path"):
        print(f"ROUTE: {prefix}{r.path}")
    elif hasattr(r, "routes"):
        for sub in r.routes:
            print_routes(sub, prefix + getattr(r, "path", ""))

for route in app.routes:
    print_routes(route)
