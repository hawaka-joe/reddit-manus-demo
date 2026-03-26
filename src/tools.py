from pydantic import BaseModel

class Tool(BaseModel):
    async def __call__(self) -> str:
        raise NotImplementedError