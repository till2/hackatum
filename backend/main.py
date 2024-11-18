from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Annotated

class DemoParams(BaseModel):
    img: bytes
    hparam1: int
    hparam2: float
    hparam3: bool

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:3000/demo",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/demo")
async def demo(picture: Annotated[str, Form()], hparam1: Annotated[int, Form()], hparam2: Annotated[float, Form()], hparam3: Annotated[int, Form()]):

    return FileResponse("cat.png")

@app.get("/demo")
async def demo():

    return FileResponse("cat.png")
