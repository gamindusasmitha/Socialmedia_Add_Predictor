import os
import math
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
import joblib
import pandas as pd
from google import genai
from google.genai import types

# Read secrets from a local .env file (never share or upload this file)
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024  # reject requests bigger than 10 KB

# Allow Vercel frontend + local testing; override via ALLOWED_ORIGINS env var on Render
DEFAULT_ORIGINS = [
    "https://mlfrontend-three.vercel.app",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]
env_origins = os.environ.get("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in env_origins.split(",") if o.strip()] or DEFAULT_ORIGINS

CORS(
    app,
    resources={r"/*": {"origins": ALLOWED_ORIGINS}},
    allow_headers=["Content-Type"],
    methods=["GET", "POST", "OPTIONS"],
)

# Stops one person from sending too many requests and using up your Gemini quota
limiter = Limiter(get_remote_address, app=app, storage_uri="memory://")

# Gemini key comes from the environment, never from the code
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Put it in the .env file or the host's environment settings.")

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(timeout=15000),  # 15 seconds per try
)

# Standard Gemini models with fallback
MODELS = ['gemini-2.5-flash', 'gemini-2.5-flash-lite']

# Load the trained ML pipeline (only load a .pkl file that you made yourself)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'social_media_addiction_model.pkl')
try:
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model_pipeline = None

# Allowed values, so users cannot send anything else
GENDERS = {'male', 'female'}
PLATFORMS = {'Instagram', 'TikTok', 'Both'}
LEVELS = {'low', 'medium', 'high'}


def number(data, key, default, low, high):
    """Read a number and check it is real and inside the allowed range."""
    try:
        value = float(data.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number")
    if not math.isfinite(value) or not (low <= value <= high):
        raise ValueError(f"{key} must be between {low} and {high}")
    return value


def choice(data, key, default, allowed):
    """Read a text value and check it is one of the allowed choices."""
    value = str(data.get(key, default))
    if value not in allowed:
        raise ValueError(f"{key} must be one of: {', '.join(sorted(allowed))}")
    return value


def ask_gemini(prompt):
    """Try each model up to 2 times. Returns text, or None if all attempts fail."""
    for model_name in MODELS:
        for attempt in range(2):
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                if getattr(response, "text", None):
                    return response.text
            except Exception as e:
                print(f"{model_name} attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
    return None


def generate_ai_recommendations(student_data, score, status):
    """Calls Gemini API to generate personalized advice based on student habits."""
    prompt = f"""
    You are a supportive student wellness and digital habits advisor.
    A student just took a social media addiction assessment and received a score of {score}/10 ({status}).

    Here are their daily habits:
- Age: {student_data['age']}
- Daily Social Media Usage: {student_data['daily_social_media_hours']} hours
- Primary Platform: {student_data['platform_usage']}
- Nightly Sleep: {student_data['sleep_hours']} hours
- Screen Time Right Before Sleep: {student_data['screen_time_before_sleep']} hours
- Current CGPA: {student_data['CGPA']}
- Physical Activity: {student_data['physical_activity']} hours/day
- Offline Social Interaction Level: {student_data['social_interaction_level']}

    Provide 3 concise, practical, and empathetic bullet-point recommendations tailored specifically to their weakest habits (e.g., if sleep is low or screen time before bed is high, address that directly). Keep the tone encouraging and the response under 150 words. Do not use markdown headers.
    """
    text = ask_gemini(prompt)
    if text:
        return text
    return "Could not generate personalized AI tips right now. Try setting daily app timers and avoiding screens 1 hour before bed."


@app.route('/', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model_pipeline is not None,
    })


@app.route('/predict', methods=['POST', 'OPTIONS'])
@limiter.limit("10 per minute")
def predict():
    if request.method == 'OPTIONS':
        return ('', 204)

    if model_pipeline is None:
        return jsonify({"success": False, "error": "The prediction service is not ready."}), 500

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "Send the answers as JSON."}), 400

    try:
        age = number(data, 'age', 20, 10, 100)
        gender = choice(data, 'gender', 'female', GENDERS)
        daily_hours = number(data, 'daily_social_media_hours', 5.0, 0, 24)
        platform = choice(data, 'platform_usage', 'Instagram', PLATFORMS)
        sleep_hours = number(data, 'sleep_hours', 7.0, 0, 24)
        screen_before_sleep = number(data, 'screen_time_before_sleep', 1.5, 0, 10)
        cgpa = number(data, 'CGPA', 3.3, 0, 4)
        physical_activity = number(data, 'physical_activity', 1.0, 0, 10)
        social_interaction = choice(data, 'social_interaction_level', 'medium', LEVELS)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    try:
        student_features = {
            'age': age,
            'gender': gender,
            'daily_social_media_hours': daily_hours,
            'platform_usage': platform,
            'sleep_hours': sleep_hours,
            'screen_time_before_sleep': screen_before_sleep,
            'CGPA': cgpa,
            'physical_activity': physical_activity,
            'social_interaction_level': social_interaction,
            'total_screen_burden': daily_hours + screen_before_sleep,
            'social_to_sleep_ratio': daily_hours / (sleep_hours + 1e-5)
        }

        input_df = pd.DataFrame([student_features])

        predicted_score = float(model_pipeline.predict(input_df)[0])
        predicted_score = round(max(1.0, min(10.0, predicted_score)), 2)

        recommendations = None
        if predicted_score < 4.0:
            status = "Low Addiction Risk (Healthy Balance)"
        elif predicted_score < 7.0:
            status = "Moderate Addiction Risk (Mind your screen time)"
            recommendations = generate_ai_recommendations(student_features, predicted_score, status)
        else:
            status = "High Addiction Risk (Consider digital detox)"
            recommendations = generate_ai_recommendations(student_features, predicted_score, status)

        return jsonify({
            "success": True,
            "predicted_addiction_score": predicted_score,
            "status": status,
            "recommendations": recommendations
        })

    except Exception as e:
        # Details go to server logs only, not to the user
        print(f"Prediction error: {e}")
        return jsonify({"success": False, "error": "Something went wrong on the server."}), 500


@app.errorhandler(429)
def too_many(e):
    return jsonify({"success": False, "error": "Too many requests. Please wait a minute and try again."}), 429


if __name__ == '__main__':
    # Debug mode is OFF by default. For local testing only, run:  set FLASK_DEBUG=1
    debug = os.environ.get("FLASK_DEBUG") == "1"
    app.run(host='127.0.0.1', port=5000, debug=debug)