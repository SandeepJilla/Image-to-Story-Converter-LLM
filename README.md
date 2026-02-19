# Image-to-Story Converter

A self-contained Flask web application that transforms images into creative stories using OpenRouter's free AI models


## Quick Start

### Prerequisites

- Python 3.8+
- OpenRouter API key (free) - [Sign up here](https://openrouter.ai/)
- HuggingFace token (optional, for audio) - [Get token here](https://huggingface.co/settings/tokens)

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set environment variables:**

**Windows:**
```cmd
set OPENROUTER_API_KEY=your_key_here
set HUGGINGFACE_API_TOKEN=your_token_here
```

**Linux/Mac:**
```bash
export OPENROUTER_API_KEY=your_key_here
export HUGGINGFACE_API_TOKEN=your_token_here
```

Or create a `.env` file:
```env
OPENROUTER_API_KEY=your_key_here
HUGGINGFACE_API_TOKEN=your_token_here
```

3. **Run the application:**
```bash
python flask_app.py
```

4. **Open your browser:**
```
http://localhost:5000
```

## How It Works

### The Pipeline

1. **Image Upload** → User drops or selects an image (JPG/PNG/WEBP)
2. **Vision Analysis** → AI describes the image scene (Gemini Flash or similar)
3. **Story Generation** → AI creates a 50-word story (Llama 3.2 or similar)




