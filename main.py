"""
FastAPI implementation for Podcastify podcast generation service.

This module provides REST endpoints for podcast generation and audio serving,
with configuration management and temporary file handling.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
import shutil
from openai import BaseModel
import yaml
from typing import Dict, Any
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
os.makedirs(TEMP_DIR, exist_ok=True)


class PodcastData(BaseModel):
    openai_key: str
    google_key: str
    elevenlabs_key: str = ""
    name: str
    tagline: str
    urls: list[str] = []


@app.post("/generate")
async def generate_podcast_endpoint(data: PodcastData):
    """"""
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

        # print(user_config)

        # Merge configurations
        conversation_config = merge_configs(base_config, user_config)

        tts_model = conversation_config.get('tts_model', base_config.get(
            'text_to_speech', {}).get('default_tts_model', 'openai'))
        # print(conversation_config)

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
        if isinstance(result, str) and os.path.isfile(result):
            filename = f"podcast_{os.urandom(8).hex()}.mp3"
            output_path = os.path.join(TEMP_DIR, filename)
            shutil.copy2(result, output_path)
            return {"audioUrl": f"/audio/{filename}"}
        elif hasattr(result, 'audio_path'):
            filename = f"podcast_{os.urandom(8).hex()}.mp3"
            output_path = os.path.join(TEMP_DIR, filename)
            shutil.copy2(result.audio_path, output_path)
            return {"audioUrl": f"/audio/{filename}"}
        else:
            raise HTTPException(
                status_code=500, detail="Invalid result format")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
