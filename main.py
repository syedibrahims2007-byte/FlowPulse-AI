import os
import json
from typing import Optional
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler

from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# ==========================================
# 1. CONFIGURATION & SERVICES INITIALIZATION
# ==========================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY","AQ.Ab8RN6IMGg-eWjgY7n77bOPOEFLkdrGipG5t6yh8AL6oNhxuNQ")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def build_user_tasks_service(authorization: Optional[str]):
    """
    Dynamically constructs the Google Tasks API client using the Bearer token
    passed in the request's Authorization header from the authenticated user.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, 
            detail="Missing or invalid Google OAuth Access Token. Please sign in with Google."
        )
    
    token = authorization.split("Bearer ")[1].strip()
    creds = Credentials(token=token)
    return build("tasks", "v1", credentials=creds)

# ==========================================
# 2. PROACTIVE NUDGER (BACKGROUND WORKER)
# ==========================================

def send_notification(message: str):
    print("\n--------------------------------------------------")
    print("🔔 PROACTIVE TASK NUDGE DETECTED")
    print("--------------------------------------------------")
    print(message)
    print("--------------------------------------------------\n")

def check_overdue_tasks_for_user(authorization: str):
    """
    Scans overdue tasks for a specific user session using their OAuth token.
    """
    try:
        service = build_user_tasks_service(authorization)
        now_iso = datetime.now(timezone.utc).isoformat()

        results = service.tasks().list(
            tasklist='@default',
            showCompleted=False,
            showHidden=False
        ).execute()

        items = results.get('items', [])
        overdue_tasks = [
            task for task in items 
            if task.get('due') and task.get('due') < now_iso
        ]

        if overdue_tasks:
            nudge_msg = "⏰ You have overdue tasks waiting for attention:\n\n"
            for task in overdue_tasks:
                nudge_msg += f"• {task.get('title')} (Due: {task.get('due')[:10]})\n"
            nudge_msg += "\nLog in to reschedule or mark them completed!"
            
            send_notification(nudge_msg)
            return overdue_tasks

    except Exception as e:
        print(f"Error checking overdue tasks: {e}")
        return []

scheduler = BackgroundScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.start()
    print("⚡ FlowPulse AI Backend Started | Multi-User Google Auth Ready")
    yield
    scheduler.shutdown()
    print("⚡ FlowPulse AI Backend Shutdown Cleanly")

app = FastAPI(title="FlowPulse AI Engine", lifespan=lifespan)

# CORS enabled for cross-origin web deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 3. REQUEST/RESPONSE MODELS & ENDPOINTS
# ==========================================

class GoalDecomposeRequest(BaseModel):
    goal: str
    tasklist_id: str = "@default"

class TaskGuideRequest(BaseModel):
    task_title: str
    task_notes: str = ""

class ArtifactRequest(BaseModel):
    task_title: str
    artifact_type: str  # 'email', 'code', or 'doc'

# Serve the Dashboard UI directly at http://localhost:8000
@app.get("/")
def read_root():
    return FileResponse("index.html")

@app.get("/api/health")
def health_check():
    return {"status": "online", "app": "FlowPulse AI Engine"}

@app.post("/decompose-goal")
async def decompose_goal(
    payload: GoalDecomposeRequest,
    authorization: Optional[str] = Header(None)
):
    try:
        service = build_user_tasks_service(authorization)

        prompt = f"""
        You are an AI project manager. Break down the following high-level goal into 4 to 6 specific, actionable sub-tasks.

        Goal: "{payload.goal}"

        Respond STRICTLY with a valid JSON array of objects, where each object has:
        - "title": Concise task name
        - "notes": Brief execution step or detail
        """

        response = gemini_client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        subtasks = json.loads(response.text)

        created_tasks = []
        for item in subtasks:
            task_body = {
                'title': item.get('title'),
                'notes': item.get('notes')
            }
            inserted = service.tasks().insert(
                tasklist=payload.tasklist_id,
                body=task_body
            ).execute()
            created_tasks.append({
                "id": inserted.get("id"),
                "title": inserted.get("title")
            })

        return {
            "status": "success",
            "original_goal": payload.goal,
            "tasks_created_count": len(created_tasks),
            "created_tasks": created_tasks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/decompose-goal-image")
async def decompose_goal_image(
    file: UploadFile = File(...),
    tasklist_id: str = Form("@default"),
    authorization: Optional[str] = Header(None)
):
    try:
        service = build_user_tasks_service(authorization)
        image_bytes = await file.read()

        prompt = """
        You are an elite project manager and visual strategist. 
        Analyze the provided image (diagram, whiteboard, sketch, or note) and:
        1. Extract the main high-level goal name.
        2. Break it down into 4 to 6 sequential sub-tasks.
        3. Generate a clean Mermaid.js flowchart string mapping task dependencies (e.g., 'graph TD; A[Task 1] --> B[Task 2];').

        Respond STRICTLY with a valid JSON object matching this schema:
        {
          "goal_title": "Extracted High-Level Goal",
          "mermaid_graph": "graph TD; A[Step 1] --> B[Step 2];",
          "subtasks": [
            {
              "title": "Concise Task Name",
              "notes": "Execution detail extracted or inferred"
            }
          ]
        }
        """

        response = gemini_client.models.generate_content(
            model='gemini-3.6-flash',
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=file.content_type
                ),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        payload = json.loads(response.text)
        created_tasks = []

        for item in payload.get("subtasks", []):
            task_body = {
                'title': item.get('title'),
                'notes': item.get('notes')
            }
            inserted = service.tasks().insert(
                tasklist=tasklist_id,
                body=task_body
            ).execute()
            created_tasks.append({
                "id": inserted.get("id"),
                "title": inserted.get("title")
            })

        return {
            "status": "success",
            "goal_title": payload.get("goal_title", "Extracted Strategy"),
            "mermaid_graph": payload.get("mermaid_graph", ""),
            "tasks_created_count": len(created_tasks),
            "created_tasks": created_tasks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/decompose-goal-audio")
async def decompose_goal_audio(
    file: UploadFile = File(...),
    tasklist_id: str = Form("@default"),
    authorization: Optional[str] = Header(None)
):
    try:
        service = build_user_tasks_service(authorization)
        audio_bytes = await file.read()

        prompt = """
        You are an elite AI assistant capable of processing spoken voice memos and audio recordings.
        Listen to the audio and:
        1. Extract or infer the main overarching goal or topic discussed.
        2. Extract 4 to 6 clear, actionable sub-tasks mentioned or implied in the speech.
        3. Generate a clean Mermaid.js flowchart string mapping the workflow dependencies (e.g., 'graph TD; A[Task 1] --> B[Task 2];').

        Respond STRICTLY with a valid JSON object matching this schema:
        {
          "goal_title": "Extracted Goal from Voice Memo",
          "mermaid_graph": "graph TD; A[Step 1] --> B[Step 2];",
          "subtasks": [
            {
              "title": "Concise Task Name",
              "notes": "Context extracted from spoken audio"
            }
          ]
        }
        """

        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=file.content_type
                ),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        payload = json.loads(response.text)
        created_tasks = []

        for item in payload.get("subtasks", []):
            task_body = {
                'title': item.get('title'),
                'notes': item.get('notes')
            }
            inserted = service.tasks().insert(
                tasklist=tasklist_id,
                body=task_body
            ).execute()
            created_tasks.append({
                "id": inserted.get("id"),
                "title": inserted.get("title")
            })

        return {
            "status": "success",
            "goal_title": payload.get("goal_title", "Voice Memo Strategy"),
            "mermaid_graph": payload.get("mermaid_graph", ""),
            "tasks_created_count": len(created_tasks),
            "created_tasks": created_tasks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-task-guide")
async def generate_task_guide(payload: TaskGuideRequest):
    try:
        prompt = f"""
        You are an expert execution coach. Provide a concise, highly practical step-by-step guide on how to complete this specific task:

        Task Title: "{payload.task_title}"
        Context/Notes: "{payload.task_notes}"

        Structure your response clearly using bullet points, key tools/links to use, and any exact code/text templates if applicable. Keep it actionable and under 250 words.
        """

        response = gemini_client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt
        )

        return {
            "status": "success",
            "task_title": payload.task_title,
            "guide": response.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-artifact")
async def generate_artifact(payload: ArtifactRequest):
    try:
        if payload.artifact_type == "email":
            prompt = f"Write a professional, ready-to-send email draft to complete or delegate this task: '{payload.task_title}'. Include Subject line and Placeholders in [brackets]."
        elif payload.artifact_type == "code":
            prompt = f"Provide a complete, production-ready code script or automation snippet to accomplish this task: '{payload.task_title}'. Include brief inline comments."
        else:
            prompt = f"Create a comprehensive document outline, brief, or specification for this task: '{payload.task_title}'. Use bullet points and clear sections."

        response = gemini_client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt
        )

        return {
            "status": "success",
            "artifact_type": payload.artifact_type,
            "content": response.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trigger-nudge-check")
async def trigger_manual_nudge_check(authorization: Optional[str] = Header(None)):
    overdue = check_overdue_tasks_for_user(authorization)
    return {
        "status": "success", 
        "message": "Overdue task scan executed.", 
        "overdue_count": len(overdue) if overdue else 0
    }
