import io
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_api import complex_chain as lang
from pydantic import BaseModel


class DemoParams(BaseModel):
    img: bytes
    hparam1: int
    hparam2: float
    hparam3: bool


class TextInput(BaseModel):
    text: str


global state
state = {}

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


@app.post("/api/demo")
async def demo(
    picture: Annotated[str, Form()],
    hparam1: Annotated[int, Form()],
    hparam2: Annotated[float, Form()],
    hparam3: Annotated[int, Form()],
):
    return FileResponse("cat.png")


@app.get("/api/demo")
async def demo():
    return FileResponse("cat.png")


@app.post("/api/lifeplanner_request")
async def lifeplanner_request(request: Request, input: TextInput):
    if request.client.host not in state:
        state[request.client.host] = lang.setup_memory()

    print(state)
    # try:
    lifestyles, facts, fact_categories, emojis, housing_facts = (
        lang.generate_lifestyles(**state[request.client.host], situation=input.text)
    )
    return {
        "lifestyles": lifestyles,
        "facts": facts,
        "fact_categories": fact_categories,
        "emojis": emojis,
    }
    # except Exception as e:
    #     return {"error": str(e)}


@app.post("/api/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """
    Endpoint to handle image upload. Currently, it echoes back the uploaded image.
    """
    print(f"Received file: {file.filename}, Content-Type: {file.content_type}")
    try:
        contents = await file.read()
        return StreamingResponse(io.BytesIO(contents), media_type=file.content_type)
    except Exception as e:
        print(f"Error processing uploaded image: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process the image.")


app.mount("/", StaticFiles(directory="static", html=True), name="frontend")
