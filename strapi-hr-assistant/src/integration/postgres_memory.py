from contextlib import asynccontextmanager

from config.settings import settings
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


_is_setup_done = False


@asynccontextmanager
async def open_memory(): 
    global _is_setup_done
    async with AsyncPostgresSaver.from_conn_string(
        settings.DATABASE_URL,
    ) as checkpointer:        
        
        async with AsyncPostgresStore.from_conn_string(
            settings.DATABASE_URL
        ) as store:
            
            if not _is_setup_done:
                await checkpointer.setup()
                await store.setup()
                _is_setup_done = True

            yield checkpointer, store