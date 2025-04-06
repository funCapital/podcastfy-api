"""
FastAPI implementation for Podcastify podcast generation service.

This module provides REST endpoints for podcast generation and audio serving,
with configuration management and temporary file handling.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import os
import shutil
import json
import uuid
import time
import threading
from openai import BaseModel
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from podcastfy.client import generate_podcast
import uvicorn

import requests


def download_content(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    return response.content.decode('utf-8')  # Decode content to UTF-8


def load_base_config() -> Dict[Any, Any]:
    config_path = Path(__file__).parent / "podcastfy" / \
        "conversation_config.yaml"
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Warning: Could not load base config: {e}")
        return {}


def merge_configs(base_config: Dict[Any, Any], user_config: Dict[Any, Any]) -> Dict[Any, Any]:
    """Merge user configuration with base configuration, preferring user values."""
    merged = base_config.copy()

    # Handle special cases for nested dictionaries
    if 'text_to_speech' in merged and 'text_to_speech' in user_config:
        merged['text_to_speech'].update(user_config.get('text_to_speech', {}))

    # Update top-level keys
    for key, value in user_config.items():
        if key != 'text_to_speech':  # Skip text_to_speech as it's handled above
            if value is not None:  # Only update if value is not None
                merged[key] = value

    return merged


app = FastAPI()

TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_audio")
TASKS_DIR = os.path.join(os.path.dirname(__file__), "tasks")
TASKS_FILE = os.path.join(TASKS_DIR, "tasks.json")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(TASKS_DIR, exist_ok=True)

# Task management lock to prevent race conditions
tasks_lock = threading.Lock()


class PodcastData(BaseModel):
    openai_key: str
    google_key: str
    elevenlabs_key: str = ""
    name: str
    tagline: str
    urls: list[str] = []


class TaskResponse(BaseModel):
    task_id: str


class TaskStatus(BaseModel):
    status: str  # "pending", "success", "failed"
    created_at: float
    completed_at: Optional[float] = None
    audioUrl: Optional[str] = None
    error: Optional[str] = None


def get_tasks() -> Dict[str, Dict]:
    """Load all tasks from the tasks.json file"""
    if not os.path.exists(TASKS_FILE):
        return {}
    
    try:
        with open(TASKS_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        # If the file is corrupted, start fresh
        return {}


def save_tasks(tasks: Dict[str, Dict]):
    """Save tasks to the tasks.json file"""
    with open(TASKS_FILE, 'w') as f:
        json.dump(tasks, f, indent=2)


def cleanup_old_tasks(tasks: Dict[str, Dict], max_tasks: int = 100) -> Dict[str, Dict]:
    """Remove old tasks if there are too many"""
    if len(tasks) <= max_tasks:
        return tasks
    
    # Sort tasks by creation time (oldest first)
    sorted_tasks = sorted(tasks.items(), key=lambda x: x[1].get('created_at', 0))
    
    # Keep only the newest max_tasks
    tasks_to_keep = dict(sorted_tasks[-max_tasks:])
    return tasks_to_keep


def update_task_status(task_id: str, status: str, audio_url: Optional[str] = None, error: Optional[str] = None):
    """Update task status in the shared tasks.json file"""
    with tasks_lock:
        tasks = get_tasks()
        
        if task_id not in tasks:
            # Initialize new task
            tasks[task_id] = {
                "status": status,
                "created_at": time.time(),
                "audioUrl": audio_url,
                "error": error
            }
        else:
            # Update existing task
            tasks[task_id]["status"] = status
            
            if status in ["success", "failed"]:
                tasks[task_id]["completed_at"] = time.time()
            
            if audio_url is not None:
                tasks[task_id]["audioUrl"] = audio_url
            
            if error is not None:
                tasks[task_id]["error"] = error
        
        # Clean up old tasks if needed
        tasks = cleanup_old_tasks(tasks)
        
        # Save updated tasks
        save_tasks(tasks)


def process_podcast_generation(task_id: str, data: PodcastData):
    try:
        # Set environment variables
        os.environ['OPENAI_API_KEY'] = data.openai_key
        os.environ['GEMINI_API_KEY'] = data.google_key
        os.environ['ELEVENLABS_API_KEY'] = data.elevenlabs_key

        # Load base configuration
        base_config = load_base_config()

        user_config = {
            'podcast_name': data.name,
            'podcast_tagline': data.tagline,
        }

        # Merge configurations
        conversation_config = merge_configs(base_config, user_config)

        tts_model = conversation_config.get('tts_model', base_config.get(
            'text_to_speech', {}).get('default_tts_model', 'openai'))

        # Example of downloading content from the provided URLs
        if data.urls:
            downloaded_contents = [download_content(url) for url in data.urls]
            combined_content = "\n\n".join(downloaded_contents)
        else:
            combined_content = ""

        # Generate podcast
        result = generate_podcast(
            urls=data.urls,
            text=combined_content,
            conversation_config=conversation_config,
            tts_model=tts_model,
            longform=False,
        )

        # Handle the result
        filename = f"podcast_{os.urandom(8).hex()}.mp3"
        output_path = os.path.join(TEMP_DIR, filename)

        if isinstance(result, str) and os.path.isfile(result):
            shutil.copy2(result, output_path)
            update_task_status(task_id, "success", f"/audio/{filename}")
        elif hasattr(result, 'audio_path'):
            shutil.copy2(result.audio_path, output_path)
            update_task_status(task_id, "success", f"/audio/{filename}")
        else:
            update_task_status(task_id, "failed", error="Invalid result format")

    except Exception as e:
        update_task_status(task_id, "failed", error=str(e))


@app.post("/generate", response_model=TaskResponse)
async def generate_podcast_endpoint(data: PodcastData, background_tasks: BackgroundTasks):
    """Start an asynchronous podcast generation task and return a task ID"""
    # Generate a unique task ID
    task_id = str(uuid.uuid4())

    # Initialize task status as pending
    update_task_status(task_id, "pending")

    # Queue the podcast generation as a background task
    background_tasks.add_task(process_podcast_generation, task_id, data)

    # Return the task ID immediately
    return {"task_id": task_id}


@app.get("/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get the current status of a podcast generation task"""
    with tasks_lock:
        tasks = get_tasks()
        
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task_data = tasks[task_id]
    
    return TaskStatus(**task_data)


@app.get("/tasks", response_model=Dict[str, TaskStatus])
async def get_all_tasks():
    """Get all tasks (for debugging/monitoring)"""
    with tasks_lock:
        tasks = get_tasks()
    
    return tasks


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """ Get File Audio From ther Server"""
    file_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


@app.get("/health")
async def healthcheck():
    return {"status": "healthy"}

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host=host, port=port)
