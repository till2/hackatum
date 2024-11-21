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

class TextInput(BaseModel):
    text: str

app = FastAPI()

origins = [
    "*",  # During dev: allow all domains
    # TODO: "https://your-production-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow requests from given domains. Use ["*"] for all domains or origins
    allow_credentials=False,  # Disable cookies and credentials
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all HTTP headers
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

@app.post("/transform_text")
async def transform_text(input: TextInput):
    print("Received request to /transform_text")
    print(f"Input data: {input}")
    try:
        output_text = input.text[::-1]
        print(f"Successfully processed. Returning: {output_text}")
        return {"output": output_text}
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise
