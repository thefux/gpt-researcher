from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import os
from gpt_researcher.utils.websocket_manager import WebSocketManager
from .utils import write_md_to_pdf

import asyncio

class ResearchRequest(BaseModel):
    task: str
    report_type: str
    agent: str


app = FastAPI()

manager = WebSocketManager()


# Dynamic directory for outputs once first research is run
@app.on_event("startup")
def startup_event():
    if not os.path.isdir("outputs"):
        os.makedirs("outputs")
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

async def run_task(task, report_type, websocket):
    report = await manager.start_streaming(task, report_type, websocket)
    path = await write_md_to_pdf(report)
    await websocket.send_json({"type": "path", "output": path})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith("start"):
                json_data = json.loads(data[6:])
                task = json_data.get("task")
                report_type = json_data.get("report_type")
                if task and report_type:
                    asyncio.create_task(run_task(task, report_type, websocket))
                else:
                    print("Error: not enough parameters provided.")

    except WebSocketDisconnect:
        await manager.disconnect(websocket)

