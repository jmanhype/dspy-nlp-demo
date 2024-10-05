# DSPy NLP Demo

This project demonstrates combined NLP capabilities using DSPy models, including entity extraction, sentiment analysis, and summarization. It's exposed as a Flask API with an HTML, JS, and CSS front end.

## Features

- Entity Extraction: Identifies entities and their relationships in the input text.
- Sentiment Analysis: Determines the sentiment (positive, negative, or neutral) of the input text.
- Summarization: Provides a concise summary of the input document.

## Technologies Used

- Python
- Flask
- DSPy
- Groq LLM
- HTML/CSS/JavaScript

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/jmanhype/dspy-nlp-demo.git
   cd dspy-nlp-demo
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory and add your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

4. Run the application:
   ```
   python poc_script_with_flask_and_frontend.py
   ```

5. Open a web browser and navigate to `http://localhost:5000` to use the application.

## Usage

1. Enter your text in the provided textarea.
2. Click the "Analyze" button.
3. View the results for entity extraction, sentiment analysis, and summarization.

## File Structure

- `poc_script_with_flask_and_frontend.py`: Main Python script containing the Flask app and NLP models.
- `templates/index.html`: HTML template for the front end.
- `static/style.css`: CSS styles for the front end.
- `static/script.js`: JavaScript for handling user interactions and API calls.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).