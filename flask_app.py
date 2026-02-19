"""
Image-to-Story Converter — Flask Backend - Uses OpenRouter free models for vision + story generation 


Usage:
    set OPENROUTER_API_KEY=your_key_here
    set HUGGINGFACE_API_TOKEN=your_token_here  (optional, for audio)
    python flaskapp.py
"""

import os
import re
import json
import base64
import time
from pathlib import Path
from flask import Flask, request, jsonify, Response, make_response, send_file
import requests

# ── Configuration ──────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
HUGGINGFACE_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN", "")

# Free models available on OpenRouter
_FALLBACK_MODELS = {
    "vision": [
        {"id": "nvidia/nemotron-nano-12b-v2-vl:free", "label": "Nvidia Nemotron Nano Vision (free)"},
        {"id": "openrouter/aurora-alpha", "label": "OpenRouter Aurora Alpha (free)"},
        {"id": "openrouter/free", "label": "OpenRouter Free (auto)"},
    ],
    "text": [
        {"id": "z-ai/glm-4.5-air:free", "label": "GLM 4.5 Air (free)"},
        {"id": "meta-llama/llama-3.2-3b-instruct:free", "label": "Llama 3.2 3B (free)"},
        {"id": "deepseek/deepseek-chat-v3-0324:free", "label": "DeepSeek V3 (free)"},
        {"id": "qwen/qwen3-235b-a22b:free", "label": "Qwen3 235B (free)"},
        {"id": "microsoft/mai-ds-r1:free", "label": "Microsoft MAI-DS-R1 (free)"},
    ]
}

FREE_MODELS = _FALLBACK_MODELS.copy()
# Using non-reasoning models by default for cleaner output
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen/qwen3-vl-30b-a3b-thinking")
STORY_MODEL = os.environ.get("STORY_MODEL", "openrouter/aurora-alpha")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HUGGINGFACE_TTS_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def get_headers():
    """Build OpenRouter headers"""
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "Image to Story Converter",
    }


def call_openrouter(payload, max_retries=3):
    """
    POST to OpenRouter with automatic retry on 429.
    Returns (response_data, used_model, error_string).
    """
    base_model = payload.get("model")
    # Better vision model detection
    is_vision_request = any(
        isinstance(msg.get("content"), list) 
        for msg in payload.get("messages", [])
    )
    model_type = "vision" if is_vision_request else "text"
    
    candidates = [base_model]
    
    # fallback models
    for m in FREE_MODELS.get(model_type, []):
        if m["id"] != base_model:
            candidates.append(m["id"])

    tried = set()
    last_error = None
    
    for candidate in candidates:
        if candidate in tried:
            continue
        tried.add(candidate)
        payload["model"] = candidate
        
        print(f"[API] Trying model: {candidate}")

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    OPENROUTER_URL,
                    headers=get_headers(),
                    json=payload,
                    timeout=120
                )
            except requests.exceptions.Timeout:
                last_error = "Request timed out after 120s"
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                break
            except Exception as e:
                last_error = str(e)
                break

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
                retry_after = min(retry_after, 10)
                last_error = f"Rate limited (429) - waiting {retry_after}s"
                print(f"[API] {last_error}")
                if attempt < max_retries - 1:
                    time.sleep(retry_after)
                    continue
                else:
                    # Try next model
                    break

            if not resp.ok:
                try:
                    err_body = resp.json()
                    err_msg = err_body.get("error", {}).get("message", "") or str(err_body)
                except Exception:
                    err_msg = resp.text or f"HTTP {resp.status_code}"
                
                last_error = f"{resp.status_code}: {err_msg}"
                print(f"[API] Error: {last_error}")
                
                if resp.status_code == 404:
                    # Model not found, try next
                    break
                elif resp.status_code == 402:
                    # Payment required, try next
                    break
                else:
                    # Other error, might be transient
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    break

            try:
                data = resp.json()
            except Exception as e:
                last_error = f"Invalid JSON response: {str(e)}"
                break
                
            if "error" in data:
                err_msg = data["error"].get("message", str(data["error"]))
                last_error = err_msg
                print(f"[API] Response error: {err_msg}")
                # Try next model
                break

            print(f"[API] Success with model: {candidate}")
            return data, candidate, None

    # All models failed
    error_msg = last_error or "All models failed"
    if "rate" in error_msg.lower() or "429" in error_msg:
        return None, base_model, "Rate limit reached on all free models. Please wait 1-2 minutes and try again, or try a different image."
    else:
        return None, base_model, f"All models failed. Last error: {error_msg}"


def fetch_free_models():
    """Fetch live free models from OpenRouter API"""
    global FREE_MODELS
    
    # Start with fallback models
    FREE_MODELS = _FALLBACK_MODELS.copy()
    
    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            timeout=10
        )
        if not resp.ok:
            print(f"[models] OpenRouter API {resp.status_code} — using fallback list")
            return
        
        data = resp.json().get("data", [])
        vision_models = []
        text_models = []
        
        for m in data:
            pricing = m.get("pricing", {})
            # Check if free (prompt price is "0")
            if str(pricing.get("prompt", "1")) != "0":
                continue
            
            mid = m.get("id", "")
            name = m.get("name", mid)
            ctx = m.get("context_length", 0)
            
            model_info = {
                "id": mid,
                "label": f"{name} [{ctx//1000}k ctx]"
            }
            
            # Check architecture field for modality
            arch = m.get("architecture", {})
            modality = arch.get("modality", "")
            
            # Vision models: modality is "text+image" or has vision keywords
            is_vision = (
                modality == "text+image" or 
                "vision" in mid.lower() or 
                "vl" in mid.lower() or
                "-v" in mid.lower()
            )
            
            if is_vision:
                vision_models.append(model_info)
                print(f"[models] Found vision model: {mid}")
            else:
                text_models.append(model_info)
        
        # Update FREE_MODELS with API results (keep fallback if empty)
        if vision_models:
            FREE_MODELS["vision"] = vision_models
            print(f"[models] Using {len(vision_models)} vision models from API")
        else:
            print(f"[models] No vision models from API, using {len(FREE_MODELS['vision'])} fallback models")
        
        if text_models:
            FREE_MODELS["text"] = text_models
            print(f"[models] Using {len(text_models)} text models from API")
        else:
            print(f"[models] No text models from API, using {len(FREE_MODELS['text'])} fallback models")
        
    except Exception as e:
        print(f"[models] Fetch failed ({e}) — using fallback list")
    
    print(f"[models] Final: {len(FREE_MODELS['vision'])} vision + {len(FREE_MODELS['text'])} text models")


def image_to_base64(image_path):
    """Convert image file to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def analyze_image(image_path):
    """Use vision model to analyze image, with text model fallback"""
    try:
        base64_image = image_to_base64(image_path)
        print(f"[Vision] Image encoded to base64, length: {len(base64_image)}")
        
        # Try vision models first
        payload = {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one detailed sentence (max 40 words), focusing on the main subject, setting, and mood. Be descriptive but concise."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 150
        }
        
        print(f"[Vision] Calling OpenRouter with model: {VISION_MODEL}")
        data, used_model, err = call_openrouter(payload)
        
        if err and ("rate" in err.lower() or "429" in err):
            print("[Vision] All vision models rate-limited")
            return None, f"Vision models are currently rate-limited. Please wait 1-2 minutes and click 'Retry Generation'."
        
        if err:
            print(f"[Vision] Error: {err}")
            return None, f"Vision error: {err}"
        
        # Debug: print the entire response
        print(f"[Vision] Full API response: {json.dumps(data, indent=2)[:500]}")
        
        # Extract content
        try:
            message = data["choices"][0]["message"]
            print(f"[Vision] Message object keys: {message.keys()}")
            
            # Try content first
            description = message.get("content", "").strip()
            
            # If content is empty, try reasoning field (some models use this)
            if not description and "reasoning" in message:
                reasoning_text = message.get("reasoning", "").strip()
                print(f"[Vision] Using reasoning field instead of content")
                
                # The reasoning field contains the model's thinking process
                # Look for the actual description at the end, often after phrases like:
                # "Here's my description:", "Final description:", or quoted text
                
                # Try to extract just the description part
                lines = reasoning_text.split('\n')
                # Often the last few lines contain the actual output
                # or look for quoted text
                
                # Check for quoted descriptions
                import re
                quotes = re.findall(r'"([^"]+)"', reasoning_text)
                if quotes:
                    # Use the longest quote as it's likely the description
                    description = max(quotes, key=len)
                    print(f"[Vision] Extracted from quotes")
                else:
                    # Otherwise, take last substantial sentence
                    sentences = [s.strip() for s in reasoning_text.split('.') if len(s.strip()) > 20]
                    if sentences:
                        description = sentences[-1] + '.'
                        print(f"[Vision] Extracted last sentence")
                    else:
                        description = reasoning_text[:300]  # Fallback to first 300 chars
            
            # If still empty, try reasoning_details
            if not description and "reasoning_details" in message:
                reasoning_details = message.get("reasoning_details", [])
                if reasoning_details and len(reasoning_details) > 0:
                    description = reasoning_details[0].get("text", "").strip()
                    print(f"[Vision] Using reasoning_details field")
                    
        except (KeyError, IndexError, TypeError) as e:
            print(f"[Vision] Failed to extract content: {e}")
            print(f"[Vision] Data structure: {data}")
            return None, f"Failed to parse API response: {str(e)}"
        
        print(f"[Vision] Raw description: '{description[:200]}...'")
        print(f"[Vision] Description type: {type(description)}")
        print(f"[Vision] Description length: {len(description)}")
        
        if not description:
            print(f"[Vision] WARNING: Empty description from API!")
            return None, "Vision model returned empty response. Try a different vision model from the dropdown."
        
        # Truncate if too long (we only need a brief description)
        if len(description) > 300:
            description = description[:300].rsplit('.', 1)[0] + '.'
        
        return description, None
        
    except Exception as e:
        print(f"[Vision] Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"Image analysis failed: {str(e)}"


def generate_story(scenario):
    """Generate story from scenario using text model"""
    try:
        print(f"[Story] Input scenario: '{scenario}'")
        print(f"[Story] Scenario length: {len(scenario)}")
        
        prompt = f"""You are a talented storyteller who creates engaging short stories.

Based on this scene description, write a captivating story in exactly 50 words or less:

SCENE: {scenario}

Write a creative, vivid story with a clear beginning and ending. Make it engaging and imaginative.

STORY:"""
        
        payload = {
            "model": STORY_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.9,
            "max_tokens": 200
        }
        
        print(f"[Story] Calling OpenRouter with model: {STORY_MODEL}")
        data, used_model, err = call_openrouter(payload)
        if err:
            print(f"[Story] Error: {err}")
            return None, f"Story generation error: {err}"
        
        # Debug: print the entire response
        print(f"[Story] Full API response: {json.dumps(data, indent=2)[:500]}")
        
        # Extract content
        try:
            message = data["choices"][0]["message"]
            print(f"[Story] Message object keys: {message.keys()}")
            
            # Try content first
            story = message.get("content", "").strip()
            
            # If content is empty, try reasoning field (some models use this)
            if not story and "reasoning" in message:
                reasoning_text = message.get("reasoning")
                if reasoning_text:  # Check if not None
                    reasoning_text = reasoning_text.strip()
                    print(f"[Story] Using reasoning field instead of content")
                    
                    # Extract the actual story from the reasoning
                    import re
                    
                    # Look for text in quotes (common pattern)
                    quotes = re.findall(r'"([^"]+)"', reasoning_text)
                    if quotes:
                        story = max(quotes, key=len)  # Use longest quote
                        print(f"[Story] Extracted from quotes")
                    else:
                        # Look for story after markers
                        markers = ["STORY:", "Story:", "Here's the story:", "Final story:", 
                                  "I'll write:", "The story:", "My story:"]
                        for marker in markers:
                            if marker in reasoning_text:
                                parts = reasoning_text.split(marker, 1)
                                if len(parts) > 1:
                                    story = parts[1].strip()
                                    # Take first paragraph/sentence
                                    story = story.split('\n\n')[0].strip()
                                    print(f"[Story] Extracted after marker: {marker}")
                                    break
                        
                        # If still no story, use last few sentences
                        if not story:
                            sentences = [s.strip() for s in reasoning_text.split('.') if len(s.strip()) > 20]
                            if len(sentences) >= 3:
                                story = '. '.join(sentences[-3:]) + '.'
                                print(f"[Story] Using last 3 sentences")
                            elif sentences:
                                story = sentences[-1] + '.'
                            else:
                                story = reasoning_text[:200]
            
            # If still empty, try reasoning_details
            if not story and "reasoning_details" in message:
                reasoning_details = message.get("reasoning_details", [])
                if reasoning_details and len(reasoning_details) > 0:
                    story = reasoning_details[0].get("text", "").strip()
                    print(f"[Story] Using reasoning_details field")
                    
        except (KeyError, IndexError, TypeError) as e:
            print(f"[Story] Failed to extract content: {e}")
            print(f"[Story] Data structure: {data}")
            return None, f"Failed to parse API response: {str(e)}"
        
        print(f"[Story] Raw story length: {len(story)}")
        
        if not story:
            print(f"[Story] WARNING: Empty story from API - using creative fallback")
            # Generate a simple creative story based on the description
            words = scenario.split()[:10]
            subject = ' '.join(words)
            story = f"In a realm where {subject}, magic stirred. Shadows danced, light flickered, and destiny awaited. The scene held secrets untold, inviting wonder and imagination to take flight."
            print(f"[Story] Using fallback story")
        
        # Extract just the story part if the model included reasoning
        # Look for common story markers
        story_markers = ["STORY:", "Story:", "Here's the story:", "Here is the story:"]
        for marker in story_markers:
            if marker in story:
                story = story.split(marker, 1)[1].strip()
                print(f"[Story] Extracted story after marker")
                break
        
        # Truncate if too long (we want max 50-100 words)
        words = story.split()
        if len(words) > 100:
            story = ' '.join(words[:100]) + '...'
            print(f"[Story] Truncated to 100 words")
        
        print(f"[Story] Final story: '{story[:200]}...'")
        return story, None
        
        if not story:
            print(f"[Story] WARNING: Empty story from API!")
            return None, "Story model returned empty response. Try again."
        
        return story, None
        
    except Exception as e:
        print(f"[Story] Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"Story generation failed: {str(e)}"


def generate_audio(text):
    """Generate audio from text using HuggingFace TTS"""
    if not HUGGINGFACE_API_TOKEN:
        return False, "HuggingFace token not configured"
    
    try:
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
        payload = {"inputs": text}
        
        response = requests.post(
            HUGGINGFACE_TTS_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            audio_path = "generated_audio.flac"
            with open(audio_path, "wb") as f:
                f.write(response.content)
            return True, None
        else:
            return False, f"TTS API returned {response.status_code}"
            
    except Exception as e:
        return False, f"Audio generation failed: {str(e)}"


# ── Embedded HTML ──────────────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Image to Story Converter</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<style>
:root {
  --bg: #04090f; --surface: #080f1a; --panel: #0c1624;
  --border: #132035; --border2: #1a2d48;
  --teal: #00d4b4; --teal-dim: #00856e;
  --blue: #1a78ff; --blue-dim: #0d3d80;
  --gold: #f0a020; --purple: #b24bf3; --red: #ff4060;
  --text: #ccd8e8; --text-dim: #526880; --text-mute: #2a3f58;
  --mono: 'Space Mono', monospace; --sans: 'DM Sans', sans-serif;
  --radius: 12px;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; overflow: hidden; background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 14px; line-height: 1.6; }

body::before {
  content: ''; position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image: linear-gradient(rgba(0,212,180,.04) 1px, transparent 1px), linear-gradient(90deg, rgba(0,212,180,.04) 1px, transparent 1px);
  background-size: 40px 40px; animation: gridDrift 30s linear infinite;
}
@keyframes gridDrift { to { background-position: 40px 40px; } }

body::after {
  content: ''; position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background: radial-gradient(ellipse 600px 400px at 10% 80%, rgba(178,75,243,.08) 0%, transparent 70%), 
              radial-gradient(ellipse 500px 300px at 90% 20%, rgba(26,120,255,.07) 0%, transparent 70%);
}

#app { position: relative; z-index: 1; display: flex; flex-direction: column; height: 100vh; }

header { display: flex; align-items: center; gap: 12px; padding: 0 24px; height: 64px;
  background: rgba(8,15,26,.95); border-bottom: 1px solid var(--border); backdrop-filter: blur(12px); }

.logo { width: 36px; height: 36px; background: conic-gradient(from 200deg, var(--purple), var(--teal), var(--purple));
  clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
  animation: rotateLogo 10s linear infinite; }
@keyframes rotateLogo { to { filter: hue-rotate(60deg); } }

header h1 { font-family: var(--mono); font-size: 14px; font-weight: 700; letter-spacing: .12em;
  text-transform: uppercase; background: linear-gradient(135deg, var(--teal), var(--purple));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

.header-sub { font-size: 11px; color: var(--text-dim); margin-left: 4px; }

.status-bar { margin-left: auto; display: flex; align-items: center; gap: 16px; }
.status-chip { display: flex; align-items: center; gap: 6px; font-family: var(--mono);
  font-size: 10px; letter-spacing: .06em; color: var(--text-dim); }
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--text-mute); transition: background .3s; }
.status-dot.ok { background: var(--teal); box-shadow: 0 0 8px var(--teal); }
.status-dot.fail { background: var(--red); box-shadow: 0 0 8px var(--red); }

main { flex: 1; display: flex; align-items: flex-start; justify-content: center; padding: 40px 20px; overflow-y: auto; }

.container { width: 100%; max-width: 900px; margin-bottom: 40px; }

.upload-section { background: var(--surface); border: 2px dashed var(--border2); border-radius: var(--radius);
  padding: 48px; text-align: center; transition: all .3s; cursor: pointer; position: relative; }
.upload-section:hover, .upload-section.drag-over { border-color: var(--teal); background: rgba(0,212,180,.03); }
.upload-section.has-image { border-style: solid; border-color: var(--teal); }

.upload-icon { font-size: 64px; margin-bottom: 16px; opacity: .6; }
.upload-text { font-size: 16px; color: var(--text); margin-bottom: 8px; font-weight: 500; }
.upload-hint { font-size: 12px; color: var(--text-dim); }
.upload-hint strong { color: var(--teal); }

#file-input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }

.preview-container { display: none; margin-top: 24px; }
.preview-container.show { display: block; animation: fadeIn .4s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(12px); } }

.image-preview { width: 100%; max-height: 400px; object-fit: contain; border-radius: var(--radius);
  box-shadow: 0 8px 24px rgba(0,0,0,.4); margin-bottom: 20px; }

.generate-btn { width: 100%; padding: 16px; background: linear-gradient(135deg, var(--teal), var(--blue));
  border: none; border-radius: var(--radius); color: white; font-family: var(--sans);
  font-size: 15px; font-weight: 600; cursor: pointer; transition: all .3s;
  display: flex; align-items: center; justify-content: center; gap: 10px; }
.generate-btn:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,212,180,.3); }
.generate-btn:disabled { background: var(--border2); cursor: not-allowed; opacity: .5; }

.results-section { display: none; margin-top: 32px; }
.results-section.show { display: block; animation: fadeIn .4s ease; }

.result-card { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 20px; margin-bottom: 16px; cursor: pointer; transition: all .3s; }
.result-card:hover { border-color: var(--teal-dim); background: rgba(0,212,180,.02); }

.result-header { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; user-select: none; }
.result-icon { font-size: 24px; }
.result-title { font-family: var(--mono); font-size: 11px; letter-spacing: .12em;
  text-transform: uppercase; color: var(--teal); }

.result-content { font-size: 14px; color: var(--text); line-height: 1.7; user-select: text; cursor: text; }

.audio-player { margin-top: 16px; width: 100%; }

audio { width: 100%; height: 40px; border-radius: 8px; }
audio::-webkit-media-controls-panel { background: var(--panel); }

.progress-bar { width: 100%; height: 4px; background: var(--border2); border-radius: 2px;
  overflow: hidden; margin: 20px 0; display: none; }
.progress-bar.show { display: block; }
.progress-fill { height: 100%; background: linear-gradient(90deg, var(--teal), var(--purple));
  width: 0%; transition: width .3s; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .6; } }

.status-message { text-align: center; font-family: var(--mono); font-size: 12px;
  color: var(--text-dim); margin: 12px 0; display: none; }
.status-message.show { display: block; }

.reset-btn { margin-top: 24px; padding: 12px 24px; background: none; border: 1px solid var(--border);
  border-radius: 8px; color: var(--text-dim); font-size: 13px; cursor: pointer;
  transition: all .3s; display: none; }
.reset-btn.show { display: inline-block; }
.reset-btn:hover { border-color: var(--red); color: var(--red); }

.error-message { background: rgba(255,64,96,.15); border: 1px solid var(--red);
  border-radius: 8px; padding: 12px 16px; color: var(--red); font-size: 13px;
  margin-top: 16px; display: none; }
.error-message.show { display: block; animation: shake .5s; }
@keyframes shake { 0%, 100% { transform: translateX(0); } 25% { transform: translateX(-8px); }
  75% { transform: translateX(8px); } }

.model-selector { display: flex; flex-direction: column; gap: 12px; padding: 20px; background: var(--surface);
  border-radius: var(--radius); margin-bottom: 24px; }

.model-group { display: flex; align-items: center; gap: 12px; }
.model-label { font-family: var(--mono); font-size: 10px; letter-spacing: .1em;
  text-transform: uppercase; color: var(--text-mute); min-width: 80px; }
.model-select { flex: 1; background: var(--panel); border: 1px solid var(--border2);
  border-radius: 8px; color: var(--text); font-family: var(--mono); font-size: 11px;
  padding: 8px 12px; cursor: pointer; outline: none; }
.model-select:hover, .model-select:focus { border-color: var(--teal); }
</style>
</head>
<body>
<div id="app">
  <header>
    <div class="logo"></div>
    <div>
      <h1>Image to Story Converter</h1>
      <div class="header-sub">AI-Powered Creative Storytelling</div>
    </div>
    <div class="status-bar">
      <div class="status-chip"><div class="status-dot" id="dot-api"></div><span>OPENROUTER</span></div>
      <div class="status-chip"><div class="status-dot" id="dot-tts"></div><span>AUDIO</span></div>
    </div>
  </header>

  <main>
    <div class="container">
      <div class="model-selector">
        <div class="model-group">
          <span class="model-label">Vision:</span>
          <select class="model-select" id="vision-model">
            <option value="">Loading models...</option>
          </select>
        </div>
        <div class="model-group">
          <span class="model-label">Story:</span>
          <select class="model-select" id="story-model">
            <option value="">Loading models...</option>
          </select>
        </div>
      </div>

      <div class="upload-section" id="upload-zone">
        <input type="file" id="file-input" accept="image/jpeg,image/jpg,image/png,image/webp"/>
        <div class="upload-icon">🖼️</div>
        <div class="upload-text">Upload an Image</div>
        <div class="upload-hint">Drop your image here or <strong>click to browse</strong><br>
        Supports JPG, PNG, WEBP (max 10MB)</div>
      </div>

      <div class="preview-container" id="preview-container">
        <img id="image-preview" class="image-preview" alt="Preview"/>
        <button class="generate-btn" id="generate-btn" onclick="generateStory()">
          <span>✨</span><span>Generate Story</span>
        </button>
      </div>

      <div class="progress-bar" id="progress-bar">
        <div class="progress-fill" id="progress-fill"></div>
      </div>
      <div class="status-message" id="status-message"></div>

      <div class="results-section" id="results-section">
        <div class="result-card" onclick="copyToClipboard('scenario-text', this)">
          <div class="result-header">
            <span class="result-icon">🔍</span>
            <span class="result-title">Image Analysis (Click to Copy)</span>
          </div>
          <div class="result-content" id="scenario-text"></div>
        </div>

        <div class="result-card" onclick="copyToClipboard('story-text', this)">
          <div class="result-header">
            <span class="result-icon">📖</span>
            <span class="result-title">Generated Story (Click to Copy)</span>
          </div>
          <div class="result-content" id="story-text"></div>
        </div>

        <div class="result-card" id="audio-card" style="display:none;">
          <div class="result-header">
            <span class="result-icon">🔊</span>
            <span class="result-title">Audio Narration</span>
          </div>
          <audio controls id="audio-player" class="audio-player"></audio>
        </div>
      </div>

      <div class="error-message" id="error-message"></div>

      <button class="reset-btn" id="reset-btn" onclick="reset()">🔄 Start Over</button>
      <button class="reset-btn" id="retry-btn" onclick="retryGeneration()" style="margin-left: 12px; display: none;">🔄 Retry Generation</button>
    </div>
  </main>
</div>

<script>
let uploadedFile = null;

const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const generateBtn = document.getElementById('generate-btn');
const resultsSection = document.getElementById('results-section');
const progressBar = document.getElementById('progress-bar');
const progressFill = document.getElementById('progress-fill');
const statusMessage = document.getElementById('status-message');
const errorMessage = document.getElementById('error-message');
const resetBtn = document.getElementById('reset-btn');
const retryBtn = document.getElementById('retry-btn');

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const files = e.dataTransfer.files;
  if (files.length > 0) handleFile(files[0]);
});

fileInput.addEventListener('change', e => {
  if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

function handleFile(file) {
  const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
  if (!validTypes.includes(file.type)) {
    showError('Please upload a valid image file (JPG, PNG, or WEBP)');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError('File size must be less than 10MB');
    return;
  }

  uploadedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    imagePreview.src = e.target.result;
    previewContainer.classList.add('show');
    uploadZone.classList.add('has-image');
    hideError();
  };
  reader.readAsDataURL(file);
}

async function generateStory() {
  if (!uploadedFile) return;

  console.log('Starting generation...');
  generateBtn.disabled = true;
  retryBtn.style.display = 'none';
  resultsSection.classList.remove('show');
  document.getElementById('audio-card').style.display = 'none';
  hideError();
  showProgress(0);
  showStatus('🎨 Analyzing image...');

  const formData = new FormData();
  formData.append('image', uploadedFile);
  formData.append('vision_model', document.getElementById('vision-model').value);
  formData.append('story_model', document.getElementById('story-model').value);

  try {
    const response = await fetch('/generate', {
      method: 'POST',
      body: formData
    });

    console.log('Response status:', response.status);

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        console.log('Stream complete');
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const dataStr = line.slice(6);
        console.log('Received data:', dataStr);
        
        try {
          const event = JSON.parse(dataStr);
          handleEvent(event);
        } catch (e) {
          console.error('Parse error:', e, 'Data:', dataStr);
        }
      }
    }
  } catch (error) {
    console.error('Generation error:', error);
    showError('Connection error: ' + error.message);
    hideProgress();
    hideStatus();
    retryBtn.style.display = 'inline-block';
  } finally {
    generateBtn.disabled = false;
  }
}

function retryGeneration() {
  generateStory();
}

function handleEvent(event) {
  console.log('Event received:', event.type, event);
  
  switch (event.type) {
    case 'progress':
      showProgress(event.progress);
      showStatus(event.message);
      break;
    case 'scenario':
      const scenarioText = document.getElementById('scenario-text');
      console.log('Setting scenario, length:', event.content?.length || 0);
      console.log('Scenario content:', event.content);
      scenarioText.textContent = event.content;
      scenarioText.style.display = 'block';
      resultsSection.classList.add('show');
      console.log('Scenario text now:', scenarioText.textContent);
      break;
    case 'story':
      const storyText = document.getElementById('story-text');
      console.log('Setting story, length:', event.content?.length || 0);
      console.log('Story content:', event.content);
      storyText.textContent = event.content;
      storyText.style.display = 'block';
      console.log('Story text now:', storyText.textContent);
      break;
    case 'audio_ready':
      const audioCard = document.getElementById('audio-card');
      const audioPlayer = document.getElementById('audio-player');
      audioPlayer.src = '/audio?t=' + Date.now();
      audioCard.style.display = 'block';
      break;
    case 'complete':
      hideProgress();
      hideStatus();
      resetBtn.classList.add('show');
      console.log('Generation complete!');
      break;
    case 'error':
      showError(event.message);
      hideProgress();
      hideStatus();
      retryBtn.style.display = 'inline-block';
      break;
  }
}

function showProgress(percent) {
  progressBar.classList.add('show');
  progressFill.style.width = percent + '%';
}

function hideProgress() {
  progressBar.classList.remove('show');
}

function showStatus(msg) {
  statusMessage.textContent = msg;
  statusMessage.classList.add('show');
}

function hideStatus() {
  statusMessage.classList.remove('show');
}

function showError(msg) {
  errorMessage.textContent = '⚠ ' + msg;
  errorMessage.classList.add('show');
}

function hideError() {
  errorMessage.classList.remove('show');
}

function reset() {
  uploadedFile = null;
  fileInput.value = '';
  imagePreview.src = '';
  previewContainer.classList.remove('show');
  resultsSection.classList.remove('show');
  uploadZone.classList.remove('has-image');
  resetBtn.classList.remove('show');
  retryBtn.style.display = 'none';
  hideError();
  hideProgress();
  hideStatus();
}

function copyToClipboard(elementId, card) {
  const element = document.getElementById(elementId);
  const text = element.textContent.trim();
  
  if (!text) {
    showError('No content to copy yet. Please generate a story first.');
    return;
  }
  
  navigator.clipboard.writeText(text).then(() => {
    // Visual feedback
    const originalBorder = card.style.borderColor;
    const originalBg = card.style.background;
    card.style.borderColor = 'var(--teal)';
    card.style.background = 'rgba(0,212,180,.08)';
    
    // Show copied message briefly
    const title = card.querySelector('.result-title');
    const originalText = title.textContent;
    title.textContent = originalText.replace('(Click to Copy)', '✓ Copied!');
    
    setTimeout(() => {
      card.style.borderColor = originalBorder;
      card.style.background = originalBg;
      title.textContent = originalText;
    }, 1500);
  }).catch(err => {
    console.error('Copy failed:', err);
    showError('Failed to copy to clipboard');
  });
}

async function loadModels() {
  try {
    console.log('Loading models...');
    const response = await fetch('/models');
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    
    console.log('Models data received:', data);
    console.log('Vision models count:', data.vision?.length || 0);
    console.log('Text models count:', data.text?.length || 0);
    
    const visionSelect = document.getElementById('vision-model');
    const storySelect = document.getElementById('story-model');
    
    if (!visionSelect || !storySelect) {
      console.error('Model select elements not found!');
      return;
    }
    
    // Clear loading text
    visionSelect.innerHTML = '';
    storySelect.innerHTML = '';
    
    // Check if we got data
    if (!data.vision || data.vision.length === 0) {
      console.error('No vision models received!');
      visionSelect.innerHTML = '<option value="">No vision models available</option>';
    } else {
      data.vision.forEach((m, index) => {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.label;
        if (m.id === data.current_vision) {
          opt.selected = true;
          console.log('Selected vision model:', m.id);
        }
        visionSelect.appendChild(opt);
        if (index < 3) console.log('Added vision model:', m.id);
      });
      console.log(`Loaded ${data.vision.length} vision models`);
    }
    
    if (!data.text || data.text.length === 0) {
      console.error('No text models received!');
      storySelect.innerHTML = '<option value="">No text models available</option>';
    } else {
      data.text.forEach((m, index) => {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.label;
        if (m.id === data.current_story) {
          opt.selected = true;
          console.log('Selected story model:', m.id);
        }
        storySelect.appendChild(opt);
        if (index < 3) console.log('Added text model:', m.id);
      });
      console.log(`Loaded ${data.text.length} text models`);
    }
    
    console.log('✅ Models loaded successfully');
  } catch (e) {
    console.error('❌ Failed to load models:', e);
    const visionSelect = document.getElementById('vision-model');
    const storySelect = document.getElementById('story-model');
    if (visionSelect) visionSelect.innerHTML = '<option value="">Error loading models</option>';
    if (storySelect) storySelect.innerHTML = '<option value="">Error loading models</option>';
    showError('Failed to load AI models. Check console for details.');
  }
}

async function checkHealth() {
  try {
    const response = await fetch('/health');
    const data = await response.json();
    document.getElementById('dot-api').className = 'status-dot ' + (data.api ? 'ok' : 'fail');
    document.getElementById('dot-tts').className = 'status-dot ' + (data.tts ? 'ok' : 'fail');
  } catch (e) {
    console.error('Health check failed:', e);
  }
}

loadModels();
checkHealth();
setInterval(checkHealth, 30000);

// Also try loading models again after a short delay in case of timing issues
setTimeout(() => {
  console.log('Retrying model load...');
  loadModels();
}, 1000);
</script>
</body>
</html>"""


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return make_response(HTML_TEMPLATE)


@app.route("/generate", methods=["POST"])
def generate():
    """Main generation endpoint with streaming progress"""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['image']
    vision_model = request.form.get('vision_model', VISION_MODEL)
    story_model = request.form.get('story_model', STORY_MODEL)
    
    if not image_file.filename:
        return jsonify({"error": "No image selected"}), 400
    
    ext = Path(image_file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400
    
    # Save uploaded image temporarily
    temp_path = f"temp_image{ext}"
    image_file.save(temp_path)
    
    def generate_stream():
        try:
            # Step 1: Analyze image
            print("[Stream] Starting image analysis...")
            yield f"data: {json.dumps({'type': 'progress', 'progress': 20, 'message': '🎨 Analyzing image...'}, ensure_ascii=False)}\n\n"
            time.sleep(0.5)
            
            global VISION_MODEL, STORY_MODEL
            VISION_MODEL = vision_model
            STORY_MODEL = story_model
            
            print(f"[Stream] Using vision model: {VISION_MODEL}")
            scenario, err = analyze_image(temp_path)
            if err:
                print(f"[Stream] Vision error: {err}")
                yield f"data: {json.dumps({'type': 'error', 'message': err}, ensure_ascii=False)}\n\n"
                return
            
            print(f"[Stream] Scenario generated: {scenario[:100]}...")
            print(f"[Stream] Scenario full length: {len(scenario)}")
            yield f"data: {json.dumps({'type': 'scenario', 'content': scenario}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'progress', 'progress': 50, 'message': '📖 Writing story...'}, ensure_ascii=False)}\n\n"
            time.sleep(0.5)
            
            # Step 2: Generate story
            print(f"[Stream] Using story model: {STORY_MODEL}")
            story, err = generate_story(scenario)
            if err:
                print(f"[Stream] Story error: {err}")
                yield f"data: {json.dumps({'type': 'error', 'message': err}, ensure_ascii=False)}\n\n"
                return
            
            print(f"[Stream] Story generated: {story[:100]}...")
            print(f"[Stream] Story full length: {len(story)}")
            yield f"data: {json.dumps({'type': 'story', 'content': story}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'progress', 'progress': 75, 'message': '🔊 Generating audio...'}, ensure_ascii=False)}\n\n"
            time.sleep(0.5)
            
            # Step 3: Generate audio
            audio_success, err = generate_audio(story)
            if audio_success:
                print("[Stream] Audio generated successfully")
                yield f"data: {json.dumps({'type': 'audio_ready'}, ensure_ascii=False)}\n\n"
            else:
                print(f"[Stream] Audio generation skipped: {err}")
            
            yield f"data: {json.dumps({'type': 'progress', 'progress': 100, 'message': '✅ Complete!'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'complete'}, ensure_ascii=False)}\n\n"
            print("[Stream] Generation complete!")
            
        except Exception as e:
            print(f"[Stream] Exception: {str(e)}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    print(f"[Stream] Cleaned up temp file: {temp_path}")
                except:
                    pass
    
    return Response(generate_stream(), content_type='text/event-stream')


@app.route("/audio")
def get_audio():
    """Serve generated audio file"""
    audio_path = "generated_audio.flac"
    if os.path.exists(audio_path):
        return send_file(audio_path, mimetype='audio/flac')
    return jsonify({"error": "Audio not found"}), 404


@app.route("/models")
def get_models():
    """Get available models"""
    print(f"[models endpoint] Returning {len(FREE_MODELS.get('vision', []))} vision, {len(FREE_MODELS.get('text', []))} text models")
    
    vision_list = FREE_MODELS.get("vision", _FALLBACK_MODELS["vision"])
    text_list = FREE_MODELS.get("text", _FALLBACK_MODELS["text"])
    
    response = {
        "vision": vision_list,
        "text": text_list,
        "current_vision": VISION_MODEL,
        "current_story": STORY_MODEL
    }
    
    print(f"[models endpoint] Sample vision model: {vision_list[0] if vision_list else 'NONE'}")
    print(f"[models endpoint] Sample text model: {text_list[0] if text_list else 'NONE'}")
    
    return jsonify(response)


@app.route("/health")
def health():
    """Health check endpoint"""
    api_ok = bool(OPENROUTER_API_KEY and len(OPENROUTER_API_KEY) > 10)
    tts_ok = bool(HUGGINGFACE_API_TOKEN and len(HUGGINGFACE_API_TOKEN) > 10)
    
    return jsonify({
        "api": api_ok,
        "tts": tts_ok,
        "vision_model": VISION_MODEL,
        "story_model": STORY_MODEL,
        "status": "ok" if api_ok else "degraded"
    })


if __name__ == "__main__":
    fetch_free_models()
    print("=" * 60)
    print("  🖼️  IMAGE TO STORY CONVERTER")
    print("=" * 60)
    print(f"  Vision Model: {VISION_MODEL}")
    print(f"  Story Model:  {STORY_MODEL}")
    key_hint = OPENROUTER_API_KEY[:12] + "..." if len(OPENROUTER_API_KEY) > 12 else "(not set)"
    print(f"  API Key:      {key_hint}")
    tts_hint = "configured" if HUGGINGFACE_API_TOKEN else "not configured (audio disabled)"
    print(f"  TTS:          {tts_hint}")
    print(f"  URL:          http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, port=5000, threaded=True)
