from flask import Flask, render_template, request, send_from_directory
import os
import gtts
import openai
from PyPDF2 import PdfReader

app = Flask(__name__)

# Configure the folder for uploaded files
UPLOAD_FOLDER = 'uploads'  # Folder where PDF and MP3 will be stored
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# OpenAI API key
openai.api_key = "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"

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle PDF file upload and text-to-speech conversion
@app.route('/upload', methods=['POST'])
def upload_pdf():
    # Check if a file is present in the form
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Extract text from the PDF
        pdf_text = extract_text_from_pdf(filename)

        # Summarize the extracted text using OpenAI GPT-3
        summarized_text = summarize_text(pdf_text)

        # Convert the summarized text to speech and save as MP3
        mp3_filename = 'output.mp3'
        mp3_path = os.path.join(app.config['UPLOAD_FOLDER'], mp3_filename)
        try:
            tts = gtts.gTTS(summarized_text)
            tts.save(mp3_path)
        except gtts.tts.gTTSError as e:
            return f"Error in TTS: {e}", 500

        # Return the result page with the MP3 file and both full and summarized text
        return render_template('result.html', text=pdf_text, summarized_text=summarized_text, mp3_filename=mp3_filename)

    return "Invalid file type", 400

# Helper function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Helper function to check if the uploaded file is a PDF
def allowed_file(filename):
    return filename.lower().endswith('.pdf')

# Helper function to summarize the extracted PDF text using OpenAI GPT-3
def summarize_text(text):
    try:
        # Use the new ChatCompletion API for summarization
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can use "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Please summarize the following text:\n\n{text}"}
            ],
            max_tokens=300,  # Limit the summary length
            temperature=0.7
        )
        summarized_text = response['choices'][0]['message']['content'].strip()
        return summarized_text
    except Exception as e:
        return f"Error in summarizing: {e}"

# Route to download the MP3 file
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=6969)
