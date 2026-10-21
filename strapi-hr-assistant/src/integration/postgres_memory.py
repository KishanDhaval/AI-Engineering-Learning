from contextlib import asynccontextmanager

from config.settings import settings
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


@asynccontextmanager
async def open_memory(): 
    async with AsyncPostgresSaver.from_conn_string(
        settings.DATABASE_URL,
    ) as checkpointer:        
        
        async with AsyncPostgresStore.from_conn_string(
            settings.DATABASE_URL
        ) as store:
            
            await checkpointer.setup()
            await store.setup()

            yield checkpointer, store