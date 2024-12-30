from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import requests
from textblob import TextBlob
from tenacity import retry, wait_exponential, stop_after_attempt
from dotenv import load_dotenv
import os
load_dotenv()


app = Flask(__name__)
CORS(app, origins=["https://ai-digital-marketing-a02df.web.app"])

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_PAGESPEED_API_KEY = os.getenv("GOOGLE_PAGESPEED_API_KEY")


# Retry logic for OpenAI API calls
@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(3))
def call_openai_api(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the AI Digital Marketing Tool!"})


# AI Content Creation
@app.route("/generate", methods=["POST"])
def generate_content():
    data = request.json
    prompt = data.get("prompt", "Write a blog about digital marketing.")
    try:
        response = call_openai_api(prompt)
        content = response.choices[0].message["content"]
        return jsonify({"content": content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# SEO Optimization
@app.route("/seo", methods=["POST"])
def seo_optimization():
    data = request.json
    url = data.get("url", "")
    if not url:
        return jsonify({"error": "URL is required"}), 400
    try:
        api_url = f"https://www.googleapis.com/pagespeedonline/v5/runPagespeed?url={url}&key={GOOGLE_PAGESPEED_API_KEY}"
        response = requests.get(api_url)
        response.raise_for_status()
        result = response.json()
        lighthouse_score = result.get("lighthouseResult", {}).get("categories", {}).get("performance", {}).get("score", 0) * 100
        return jsonify({
            "url": url,
            "performance_score": lighthouse_score
        })
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

# Ad Campaign Generator
@app.route("/ad-campaign", methods=["POST"])
def generate_ad_campaign():
    data = request.json
    product = data.get("product", "")
    audience = data.get("audience", "")
    goal = data.get("goal", "")
    if not product or not audience or not goal:
        return jsonify({"error": "Missing required fields"}), 400
    try:
        prompt = f"Generate an ad campaign for {product} targeting {audience} to {goal}."
        response = call_openai_api(prompt)
        ad_copy = response.choices[0].message["content"]
        return jsonify({"ad_copy": ad_copy})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# AI Chatbot
@app.route("/chatbot", methods=["POST"])
def ai_chatbot():
    data = request.json
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
    try:
        response = call_openai_api(user_message)
        bot_reply = response.choices[0].message["content"]
        return jsonify({"reply": bot_reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Predictive Customer Lifetime Value (CLV)
@app.route("/clv", methods=["POST"])
def predict_clv():
    data = request.json
    customer = data.get("customer", {})
    if not customer:
        return jsonify({"error": "Customer data is required"}), 400
    revenue = customer.get("revenue", 0)
    frequency = customer.get("frequency", 1)
    retention_rate = customer.get("retention_rate", 0.8)
    try:
        clv = revenue * frequency * retention_rate / (1 - retention_rate)
        return jsonify({"clv": round(clv, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Sentiment Analysis
@app.route("/sentiment", methods=["POST"])
def sentiment_analysis():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Text is required"}), 400
    try:
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        return jsonify({"sentiment": sentiment, "text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

