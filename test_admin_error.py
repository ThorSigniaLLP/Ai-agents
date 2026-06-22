import asyncio
import sys
import redis.asyncio as aioredis
sys.modules['aioredis'] = aioredis

from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

try:
    with TestClient(app) as client:
        response = client.get("/admin/login")
        print(response.status_code)
        print(response.text)
except Exception as e:
    import traceback
    traceback.print_exc()
