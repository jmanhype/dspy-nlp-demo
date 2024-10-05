# poc_script_with_flask_and_frontend.py
# POC script to demonstrate combined NLP capabilities using DSPy models: entity extraction, sentiment analysis, and summarization, now exposed as a Flask API with an HTML, JS, and CSS front end.

import os
import traceback
from dotenv import load_dotenv
import dspy
from flask import Flask, request, jsonify, render_template
from typing import List
from dslmodel import DSLModel
from pydantic import Field
import re

# Load environment variables
load_dotenv()

# Configure the LLM (Groq in this case)
llm = dspy.GROQ(
    model='mixtral-8x7b-32768',  # or another appropriate Groq model
    api_key=os.environ['GROQ_API_KEY'],
    max_tokens=2000
)

# Configure the settings for DSPy to use the language model (LLM)
dspy.settings.configure(lm=llm)

# Initialize Flask app
app = Flask(__name__)

# Entity Extraction Model (DSPy Signature)
class EntityExtractorSignature(dspy.Signature):
    """Extract entities and their relationships from the text."""
    text = dspy.InputField()
    entities = dspy.OutputField(desc="List of entities and their types")
    relationships = dspy.OutputField(desc="Relationships between the entities")

# Entity Extraction Model (DSLModel)
class EntityExtractorModel(DSLModel):
    entities: List[str] = Field(default_factory=list, description="List of entities and their types")
    relationships: List[str] = Field(default_factory=list, description="Relationships between the entities")

    @classmethod
    def parse_output(cls, entities, relationships):
        if isinstance(entities, str):
            entities = [e.strip() for e in entities.split('\n') if e.strip()]
        if isinstance(relationships, str):
            relationships = [r.strip() for r in relationships.split('\n') if r.strip()]
        return cls(entities=entities or [], relationships=relationships or [])

# Sentiment Analysis Model (DSPy Signature)
class SentimentAnalyzerSignature(dspy.Signature):
    """Analyze the sentiment of the text."""
    text = dspy.InputField()
    sentiment = dspy.OutputField(desc="The sentiment of the document (positive, negative, or neutral)")
    confidence = dspy.OutputField(desc="The confidence score of the sentiment analysis (0-1)")

# Sentiment Analysis Model (DSLModel)
class SentimentAnalyzerModel(DSLModel):
    sentiment: str = Field("", description="The sentiment of the document (positive, negative, or neutral)")
    confidence: float = Field(0.0, description="The confidence score of the sentiment analysis (0-1)")

    @classmethod
    def parse_output(cls, sentiment, confidence):
        if isinstance(confidence, str):
            confidence_match = re.search(r'\d+(\.\d+)?', confidence)
            confidence = float(confidence_match.group()) if confidence_match else 0.0
        return cls(sentiment=sentiment, confidence=confidence)

# Summarization Model (DSPy Signature)
class SummarizerSignature(dspy.Signature):
    """Summarize the document."""
    document = dspy.InputField()
    summary = dspy.OutputField(desc="10 words or less summary")

# Summarization Model (DSLModel)
class SummarizerModel(DSLModel):
    summary: str = Field("", description="10 words or less summary")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_document():
    data = request.get_json()
    document = data.get('document', '').strip()

    if not document or len(document.split()) < 5:
        return jsonify({'error': 'Please provide a more substantial text for analysis (at least 5 words).'}), 400

    response = {}

    try:
        # Entity Extraction
        try:
            entity_extractor = dspy.Predict(EntityExtractorSignature)
            entity_result = entity_extractor(text=document)
            entity_model = EntityExtractorModel.parse_output(entity_result.entities, entity_result.relationships)
            response["entity_extraction"] = {
                "entities": entity_model.entities,
                "relationships": entity_model.relationships
            }
        except Exception as e:
            app.logger.error(f"Entity extraction failed: {str(e)}")
            response["entity_extraction"] = {"error": f"Entity extraction failed: {str(e)}"}

        # Sentiment Analysis
        try:
            sentiment_analyzer = dspy.Predict(SentimentAnalyzerSignature)
            sentiment_result = sentiment_analyzer(text=document)
            sentiment_model = SentimentAnalyzerModel.parse_output(sentiment_result.sentiment, sentiment_result.confidence)
            response["sentiment_analysis"] = {
                "sentiment": sentiment_model.sentiment,
                "confidence": sentiment_model.confidence
            }
        except Exception as e:
            app.logger.error(f"Sentiment analysis failed: {str(e)}")
            response["sentiment_analysis"] = {"error": f"Sentiment analysis failed: {str(e)}"}

        # Summarization
        try:
            summarizer = dspy.Predict(SummarizerSignature)
            summary_result = summarizer(document=document)
            summary_model = SummarizerModel(summary=summary_result.summary)
            response["summarization"] = {
                "summary": summary_model.summary
            }
        except Exception as e:
            app.logger.error(f"Summarization failed: {str(e)}")
            response["summarization"] = {"error": f"Summarization failed: {str(e)}"}

    except Exception as e:
        app.logger.error(f"An error occurred during analysis: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': f'An error occurred during analysis: {str(e)}. Please try again with a different input.'}), 500

    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

# HTML, JavaScript, and CSS files to create a front end for the Flask API

# templates/index.html
index_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Analyzer</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <h1>Document Analyzer</h1>
        <textarea id="document" placeholder="Enter your document here..."></textarea>
        <button id="analyzeButton">Analyze</button>
        <div id="results"></div>
    </div>
    <script src="/static/script.js"></script>
</body>
</html>
'''

# static/style.css
style_css = '''
body {
    font-family: Arial, sans-serif;
    background-color: #f0f0f0;
    margin: 0;
    padding: 0;
}
.container {
    width: 50%;
    margin: 100px auto;
    background-color: #fff;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}
h1 {
    text-align: center;
}
textarea {
    width: 100%;
    height: 150px;
    margin-bottom: 20px;
    padding: 10px;
    border-radius: 4px;
    border: 1px solid #ccc;
}
button {
    width: 100%;
    padding: 10px;
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}
button:hover {
    background-color: #0056b3;
}
#results {
    margin-top: 20px;
}
'''

# static/script.js
script_js = '''
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('analyzeButton').addEventListener('click', function() {
        const documentInput = document.getElementById('document').value;
        fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ document: documentInput })
        })
        .then(response => response.json())
        .then(data => {
            let results = `<h3>Analysis Results:</h3>`;
            if (data.error) {
                results += `<p>${data.error}</p>`;
            } else {
                results += `<h4>Entity Extraction:</h4>`;
                results += `<p>Entities: ${data.entity_extraction.entities}</p>`;
                results += `<p>Relationships: ${data.entity_extraction.relationships}</p>`;
                results += `<h4>Sentiment Analysis:</h4>`;
                results += `<p>Sentiment: ${data.sentiment_analysis.sentiment}</p>`;
                results += `<p>Confidence: ${data.sentiment_analysis.confidence}</p>`;
                results += `<h4>Summarization:</h4>`;
                results += `<p>Summary: ${data.summarization.summary}</p>`;
            }
            document.getElementById('results').innerHTML = results;
        })
        .catch(error => {
            document.getElementById('results').innerHTML = `<p>Error: ${error}</p>`;
        });
    });
});
'''

# Write HTML, CSS, and JS to appropriate files if not present
os.makedirs('templates', exist_ok=True)
with open('templates/index.html', 'w') as f:
    f.write(index_html)

os.makedirs('static', exist_ok=True)
with open('static/style.css', 'w') as f:
    f.write(style_css)

with open('static/script.js', 'w') as f:
    f.write(script_js)