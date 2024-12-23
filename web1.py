from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
from uuid import uuid4

app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Link Form</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <h1>Enter Your Video</h1>
        <form action="/process" method="post" enctype="multipart/form-data">
            <label for="video-link">Select Video:</label>
            <input type="file" id="video-file" name="video-file" required>
            <div class="buttons">
                <button type="submit" class="btn">Upload</button>
                <button type="reset" class="btn reset">Reset</button>
            </div>
        </form>
    </div>
</body>
</html>
"""


@app.post("/process", response_class=HTMLResponse)
async def process_video(video_file: UploadFile = File(...)):
    
    if not video_file.content_type.startswith("video/mp4"):
        return HTMLResponse(content="<h1>Invalid file format. Please upload a valid video file.</h1>", status_code=400)

    video_filename = "Video"
    video_path = os.path.join("static", video_filename)

    os.makedirs("static", exist_ok=True)

    with open(video_path, "wb") as file:
        shutil.copyfileobj(video_file.file, file)

    pdf_path = os.path.join("static", "result.pdf")
    with open(pdf_path, "wb") as pdf_file:
        pdf_file.write("%PDF-1.4 Example PDF Content")

    with open("result.html", "r") as result_page:
        return result_page.read()


@app.get("/download-pdf")
async def download_pdf():
    pdf_path = os.path.join("static", "result.pdf")
    if os.path.exists(pdf_path):
        return FileResponse(pdf_path, media_type="application/pdf", filename="result.pdf")
    else:
        return HTMLResponse(content="<h1>PDF file not found.</h1>", status_code=404)
