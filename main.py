import os
import json
from typing import Optional
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form, Header
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# ==========================================
# 1. HELPER FUNCTIONS & AUTH CONTEXT
# ==========================================

def get_gemini_client() -> genai.Client:
    """
    Instantiates an isolated Gemini client.
    Configures client transport explicitly to prevent capturing user OAuth 
    Bearer headers from FastAPI request contexts.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, 
            detail="GEMINI_API_KEY environment variable is not configured on the server."
        )
    
    clean_key = api_key.strip().strip("'").strip('"')
    
    # Enforce API Key mode explicitly via HttpOptions
    return genai.Client(
        api_key=clean_key,
        http_options=types.HttpOptions(
            headers={
                "x-goog-api-key": clean_key,
                "Authorization": ""  # Explicitly clear any inherited OAuth Bearer header
            }
        )
    )

# Exponential backoff handler for Gemini 503 high-demand errors
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def call_gemini_with_retry(ai_client: genai.Client, model: str, contents, config=None):
    """
    Wraps generate_content in an exponential backoff retry loop to automatically
    recover from temporary 503 UNAVAILABLE service spikes.
    """
    return ai_client.models.generate_content(
        model=model,
        contents=contents,
        config=config
    )

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

@app.get("/")
def read_root():
    return FileResponse("index.html")

@app.get("/sitemap.xml", response_class=Response)
def get_sitemap():
    """
    Serves the dynamic sitemap XML required by Google Search Console.
    """
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://flowpulse-ai-m0v4.onrender.com/</loc>
    <lastmod>2026-09-06</lastmod>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>https://flowpulse-ai-m0v4.onrender.com/privacy</loc>
    <lastmod>2026-09-06</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.5</priority>
  </url>
  <url>
    <loc>https://flowpulse-ai-m0v4.onrender.com/terms</loc>
    <lastmod>2026-09-06</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.5</priority>
  </url>
</urlset>"""
    return Response(content=xml_content, media_type="application/xml")

@app.get("/privacy", response_class=HTMLResponse)
async def privacy_policy():
    """
    Serves the required public Privacy Policy page to pass Google Auth Branding Verification.
    """
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Privacy Policy - FlowPulse AI</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; line-height: 1.6; color: #334155; }
            h1, h2 { color: #0f172a; }
            a { color: #0066ff; }
        </style>
    </head>
    <body>
        <h1>Privacy Policy for FlowPulse AI</h1>
        <p><em>Last updated: September 2026</em></p>
        
        <h2>1. Overview</h2>
        <p>FlowPulse AI helps users automate goal breakdown, task scheduling, and workflow strategy visualization. We take user privacy and data authorization seriously.</p>
        
        <h2>2. Information We Collect and Scope Usage</h2>
        <p>Our application requests access to Google Tasks API scopes exclusively to read and insert generated sub-tasks into your default Google Tasks list at your direct command.</p>
        
        <h2>3. How We Process Data</h2>
        <p>Task content and prompt details are sent securely to Google's Gemini models for structural decomposition. We do not permanently store your personal tasks, audio recordings, or credentials on external databases.</p>
        
        <h2>4. Data Retention & Security</h2>
        <p>OAuth tokens remain in client-side memory and are sent securely via TLS headers. Tokens are never logged or stored server-side.</p>

        <h2>5. Contact Information</h2>
        <p>If you have questions regarding this policy, contact the developer at <a href="mailto:syedibrahims2007@gmail.com">syedibrahims2007@gmail.com</a>.</p>
    </body>
    </html>
    """

@app.get("/terms", response_class=HTMLResponse)
async def terms_of_service():
    """
    Serves the required public Terms of Service page to pass Google Auth Branding Verification.
    """
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Terms of Service - FlowPulse AI</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; line-height: 1.6; color: #334155; }
            h1, h2 { color: #0f172a; }
            a { color: #0066ff; }
        </style>
    </head>
    <body>
        <h1>Terms of Service for FlowPulse AI</h1>
        <p><em>Last updated: September 2026</em></p>
        
        <h2>1. Acceptance of Terms</h2>
        <p>By accessing or using FlowPulse AI, you agree to be bound by these Terms of Service. If you do not agree to these terms, please do not use the application.</p>
        
        <h2>2. Description of Service</h2>
        <p>FlowPulse AI provides AI-assisted goal decomposition, workflow visualization, and task scheduling tools integrated with Google Tasks.</p>
        
        <h2>3. User Responsibilities</h2>
        <p>You are responsible for maintaining the security of your Google account and credentials. You agree not to use the service for any unlawful activities or to upload harmful content.</p>
        
        <h2>4. Disclaimer of Warranties</h2>
        <p>FlowPulse AI is provided "as is" and "as available" without warranties of any kind, whether express or implied. We do not guarantee uninterrupted access or error-free performance.</p>
        
        <h2>5. Limitation of Liability</h2>
        <p>In no event shall FlowPulse AI or its developers be liable for any direct indirect, incidental, or consequential damages resulting from the use or inability to use the service.</p>
        
        <h2>6. Contact Us</h2>
        <p>If you have any questions regarding these Terms, contact us at <a href="mailto:syedibrahims2007@gmail.com">syedibrahims2007@gmail.com</a>.</p>
    </body>
    </html>
    """

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
        ai_client = get_gemini_client()

        prompt = f"""
        You are an AI project manager. Break down the following high-level goal into 4 to 6 specific, actionable sub-tasks.

        Goal: "{payload.goal}"

        Respond STRICTLY with a valid JSON array of objects, where each object has:
        - "title": Concise task name
        - "notes": Brief execution step or detail
        """

        response = call_gemini_with_retry(
            ai_client=ai_client,
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
        ai_client = get_gemini_client()

        image_bytes = await file.read()
        mime_type = file.content_type if file.content_type else "image/jpeg"

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

        response = call_gemini_with_retry(
            ai_client=ai_client,
            model='gemini-3.6-flash',
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type
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
        ai_client = get_gemini_client()

        audio_bytes = await file.read()
        mime_type = file.content_type if file.content_type else "audio/mp3"

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

        response = call_gemini_with_retry(
            ai_client=ai_client,
            model='gemini-3.6-flash',
            contents=[
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=mime_type
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
        ai_client = get_gemini_client()
        prompt = f"""
        You are an expert execution coach. Provide a concise, highly practical step-by-step guide on how to complete this specific task:

        Task Title: "{payload.task_title}"
        Context/Notes: "{payload.task_notes}"

        Structure your response clearly using bullet points, key tools/links to use, and any exact code/text templates if applicable. Keep it actionable and under 250 words.
        """

        response = call_gemini_with_retry(
            ai_client=ai_client,
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
        ai_client = get_gemini_client()
        if payload.artifact_type == "email":
            prompt = f"Write a professional, ready-to-send email draft to complete or delegate this task: '{payload.task_title}'. Include Subject line and Placeholders in [brackets]."
        elif payload.artifact_type == "code":
            prompt = f"Provide a complete, production-ready code script or automation snippet to accomplish this task: '{payload.task_title}'. Include brief inline comments."
        else:
            prompt = f"Create a comprehensive document outline, brief, or specification for this task: '{payload.task_title}'. Use bullet points and clear sections."

        response = call_gemini_with_retry(
            ai_client=ai_client,
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
