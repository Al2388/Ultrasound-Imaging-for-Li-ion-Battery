from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os

from scanner_service import CScanService

app = FastAPI()
scanner = CScanService()
templates = Jinja2Templates(directory="templates")

app.mount("/local", StaticFiles(directory="cscan_out"), name="local")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/start")
async def start_scan(payload: dict):
    success, msg = scanner.start_scan(payload)
    return {"success": success, "msg": msg}

@app.post("/api/stop")
async def stop_scan():
    if scanner.stop_scan(): return {"msg": "Stopping..."}
    return {"msg": "Not running"}

@app.post("/api/return")
async def return_home():
    success, msg = scanner.return_to_start()
    return {"success": success, "msg": msg}

@app.post("/api/jog_z")
async def jog_z(payload: dict):
    dist = payload.get("z", 0.0)
    success, msg = scanner.jog_z_axis(dist)
    return {"success": success, "msg": msg}

@app.get("/api/status")
async def status():
    return {
        "status": scanner.status,
        "progress": scanner.progress,
        "images": scanner.images 
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)