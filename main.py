"""
FastAPI implementation for Podcastify podcast generation service.

This module provides REST endpoints for podcast generation and audio serving,
with configuration management and temporary file handling.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse
import os
import shutil
from enum import Enum
from openai import BaseModel
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from podcastfy.client import generate_podcast
import uvicorn
from dotenv import load_dotenv

load_dotenv()
import requests

MAIN_API_SERVER = os.getenv("MAIN_API_SERVER")

def download_content(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    return response.content.decode('utf-8')  # Decode content to UTF-8


def load_base_config() -> Dict[Any, Any]:
    config_path = Path(__file__).parent / \
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
os.makedirs(TEMP_DIR, exist_ok=True)


class PodcastStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


class PodcastStatusRequest(BaseModel):
    podcastId: str
    status: PodcastStatus
    error: Optional[str] = None


class PodcastData(BaseModel):
    openai_key: str
    google_key: str
    elevenlabs_key: str = ""
    tagline: str = ""
    # webhook url to send the status of the podcast when done
    webhook_url: str
    urls: list[str] = []
    podcast_id: str = f"podcast_{os.urandom(8).hex()}"


class TaskResponse(BaseModel):
    task_id: str


def process_podcast_generation(task_id: str, data: PodcastData):
    try:
        # Set environment variables
        os.environ['OPENAI_API_KEY'] = data.openai_key
        os.environ['GEMINI_API_KEY'] = data.google_key
        os.environ['ELEVENLABS_API_KEY'] = data.elevenlabs_key

        # Load base configuration
        base_config = load_base_config()

        user_config = {
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
        transcript_path = generate_podcast(
            urls=data.urls,
            text=combined_content,
            conversation_config=conversation_config,
            tts_model=tts_model,
            longform=False,
            transcript_only=True,
        )
        
        with open(os.path.join(TEMP_DIR, f"{task_id}.txt"), "w") as f:
            f.write(transcript_path)


        # Generate podcast
        result = generate_podcast(
            transcript_file=transcript_path,
            conversation_config=conversation_config,
            tts_model=tts_model,
            longform=False,
        )

        # Handle the result
        filename = f"{task_id}.mp3"
        output_path = os.path.join(TEMP_DIR, filename)

        status_data = {
            "podcastId": task_id,
            "status": PodcastStatus.SUCCESS if result else PodcastStatus.FAILED,
            "error": None
        }

        if isinstance(result, str) and os.path.isfile(result):
            shutil.copy2(result, output_path)
        elif hasattr(result, 'audio_path'):
            shutil.copy2(result.audio_path, output_path)
        else:
            status_data["status"] = PodcastStatus.FAILED
            status_data["error"] = "Invalid result format"

        # Call the update_podcast endpoint with the status
        print(f"Calling {data.webhook_url} with status: {status_data}")
        print(requests.post(f"{data.webhook_url}", json=status_data).text)

    except Exception as e:
        # Send error status to the update endpoint
        status_data = {
            "podcastId": task_id,
            "status": PodcastStatus.FAILED,
            "error": str(e)
        }
        print(f"Calling {data.webhook_url} with status: {status_data}")
        print(requests.post(f"{data.webhook_url}", json=status_data).text)


@app.post("/generate", response_model=TaskResponse)
async def generate_podcast_endpoint(data: PodcastData, background_tasks: BackgroundTasks):
    """Start an asynchronous podcast generation task and return a task ID"""
    # Generate a unique task ID
    task_id = data.podcast_id

    # Queue the podcast generation as a background task
    background_tasks.add_task(process_podcast_generation, task_id, data)

    # Return the task ID immediately
    return {"task_id": task_id}


@app.post("/update_podcast")
async def update_podcast_status(data: PodcastStatusRequest):
    """Receive status updates for podcast generation"""
    # This endpoint would be called by your system when a podcast
    # generation succeeds or fails
    
    # You could implement any processing logic here
    # but for now just return acknowledgment
    return {"received": True, "podcast_id": data.podcast_id, "status": data.status}


@app.get("/podcasts/{podcast_id}")
async def serve_audio(podcast_id: str):
    """ Get File Audio From ther Server"""
    file_path = os.path.join(TEMP_DIR, podcast_id + ".mp3")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


@app.get("/transcripts/{podcast_id}")
async def serve_transcript(podcast_id: str):
    """ Get Transcript From the Server"""
    reference_file_path = os.path.join(TEMP_DIR, f"{podcast_id}.txt")
    
    if not os.path.exists(reference_file_path):
        raise HTTPException(status_code=404, detail="Transcript reference not found")
    
    # Read the file path from the reference file
    with open(reference_file_path, "r") as file:
        actual_transcript_path = file.read().strip()
    
    if not os.path.exists(actual_transcript_path):
        raise HTTPException(status_code=404, detail="Transcript file not found")
    
    return FileResponse(actual_transcript_path)

@app.get("/health")
async def healthcheck():
    return {"status": "healthy"}


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host=host, port=port)
