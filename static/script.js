
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
