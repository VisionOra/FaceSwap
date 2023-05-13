import os, sys

sys.path.append("SimSwap/")
import shutil

import tempfile
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import random
from SimSwap.main import face_swap


app = FastAPI()


@app.post("/upload_files/")
async def upload_files(video: UploadFile = File(...), image: UploadFile = File(...)):
    # Save the uploaded files to the server
    # Make tmporary folder if it doesn't exist
    tmpdirname = f"/tmp/{os.getpid()}_{random.randint(0, 100000)}"
    os.makedirs(tmpdirname, exist_ok=True)
    video_name = video.filename
    input_video = f"{tmpdirname}/{video_name}"
    input_image = f"{tmpdirname}/{image.filename}"
    with open(input_video, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    with open(input_image, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Make dir if it doesn't exist
    os.makedirs(f"{tmpdirname}/swaped", exist_ok=True)
    ouput_video = f"{tmpdirname}/swaped/{video_name}"

    face_swap(input=input_video, output=ouput_video, swap_image=input_image)

    # Remove image and input
    os.remove(input_video)
    os.remove(input_image)
    # Return the image file as a downloadable response
    return FileResponse(
        path=ouput_video,
        filename=video_name,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={video_name}"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
