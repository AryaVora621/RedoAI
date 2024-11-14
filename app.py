from flask import Flask, request, render_template, send_from_directory
from transformers import pipeline, BartTokenizer
from gtts import gTTS
import os
import fitz  # PyMuPDF
import re

app = Flask(__name__)

# Path for saving files
UPLOAD_FOLDER = 'uploads'
MP3_FOLDER = 'static/mp3'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MP3_FOLDER, exist_ok=True)

# Load the summarizer model and tokenizer
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

# Max token length for DistilBART model (approx. 1024 tokens)
MAX_INPUT_TOKENS = 1024

# Home route for file upload form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle PDF file upload and processing
@app.route('/upload', methods=['POST'])
def upload_pdf():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    # Save the uploaded PDF file
    pdf_filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(pdf_filename)

    # Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf(pdf_filename)

    # Split the text into manageable chunks if it exceeds MAX_INPUT_TOKENS
    text_chunks = split_text_into_chunks(pdf_text)

    # Summarize each chunk and combine the results
    full_summary = ""
    for chunk in text_chunks:
        # Check token length before summarizing
        tokenized_chunk = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
        if len(tokenized_chunk['input_ids'][0]) > MAX_INPUT_TOKENS:
            return f"Chunk exceeds maximum token limit: {len(tokenized_chunk['input_ids'][0])} tokens."

        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
        full_summary += summary[0]['summary_text'] + " "

    # Create the MP3 file from the combined summary text
    mp3_filename = f'{os.path.splitext(file.filename)[0]}_summary.mp3'
    mp3_path = os.path.join(MP3_FOLDER, mp3_filename)
    tts = gTTS(full_summary)
    tts.save(mp3_path)

    # Return the result page with the summary and mp3 download link
    return render_template('result.html', text=full_summary, mp3_file=mp3_filename)

# Extract text from a PDF using PyMuPDF
def extract_text_from_pdf(pdf_filename):
    doc = fitz.open(pdf_filename)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Split large text into smaller chunks based on the model's token limit
def split_text_into_chunks(text):
    # Split the text by sentences using regex
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Check the token length of the current chunk + next sentence
        potential_chunk = current_chunk + " " + sentence
        tokenized_potential_chunk = tokenizer(potential_chunk, return_tensors="pt", truncation=True, padding=True)
        
        if len(tokenized_potential_chunk['input_ids'][0]) <= MAX_INPUT_TOKENS:
            current_chunk = potential_chunk
        else:
            # If it exceeds, add the current chunk to the list and start a new one
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Route to download the generated MP3 file
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(MP3_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=6969)
