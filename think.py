import base64
import requests

def encode_image(image_path):
    if image_path is None:
        return None

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_image_with_query(query, encoded_image):
    if not encoded_image:
        return "Please upload an image for analysis."
    
    try:
        # For Ollama with vision support
        payload = {
            "model": "llava",
            "prompt": f"You are a medical assistant. Analyze this image and answer the following question: {query}",
            "images": [encoded_image],
            "stream": False
        }
        
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if "response" in result:
            return result["response"]
        
        return str(result)
    except requests.exceptions.ConnectionError:
        return "Error: Ollama is not running. Please:\n1. Download Ollama from https://ollama.ai\n2. Run: ollama pull llava\n3. Ollama will start automatically"
    except Exception as e:
        return f"Error analyzing image: {str(e)}"