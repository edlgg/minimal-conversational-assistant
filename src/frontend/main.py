import os
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, Depends, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from backend.assistants.assistants_catalog import assistants_catalog

current_directory = Path(__file__).parent

app = FastAPI()

app.mount(
    "/static", StaticFiles(directory=f"{current_directory}/static"), name="static"
)

templates_directory = current_directory / "templates"
templates = Jinja2Templates(directory=str(templates_directory))

@app.get("/")
async def login(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/user_message")
async def user_message(request: Request, message: str = Form(...)) -> HTMLResponse:
    data = {
        "message": message
    }
    return templates.TemplateResponse("user_message.html", {"request": request, "data": data})

@app.post("/assistant_response")
async def chat(request: Request, message) -> HTMLResponse:
    assistant_keys = list(assistants_catalog.keys())
    assistant = assistants_catalog[assistant_keys[0]]
    response = await assistant().call("42", message)
    data = {
        "message": response
    }
    return templates.TemplateResponse("assistant_response.html", {"request": request, "data": data})


if __name__ == "__main__":
    import uvicorn

    filename = os.path.basename(__file__)
    filename = os.path.splitext(filename)[0]
    uvicorn.run(f"{filename}:app", host="0.0.0.0", port=8000, reload=True, reload_includes=[str(current_directory/"..")])

