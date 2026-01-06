import os
import zipfile
import shutil
import uuid

import cv2
import numpy as np
import torch

from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.cluster import DBSCAN
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# -----------------------------
# App setup
# -----------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

torch.set_grad_enabled(False)

device = "cpu"
detector = MTCNN(keep_all=True, device=device)
embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# -----------------------------
# Helpers
# -----------------------------
def resize_for_detection(image, max_size=800):
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image, scale


# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_zip(file: UploadFile = File(...)):
    work_id = str(uuid.uuid4())
    base_dir = f"work/{work_id}"
    os.makedirs(base_dir, exist_ok=True)

    zip_path = os.path.join(base_dir, "input.zip")
    with open(zip_path, "wb") as f:
        f.write(await file.read())

    extract_dir = os.path.join(base_dir, "images")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    embeddings = []
    image_names = []

    no_face_images = []

    # -----------------------------
    # ONE embedding per image
    # -----------------------------
    for img_name in os.listdir(extract_dir):
        if not img_name.lower().endswith(IMAGE_EXTS):
            continue

        img_path = os.path.join(extract_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        small, scale = resize_for_detection(image)
        boxes, _ = detector.detect(small)

        if boxes is None:
            no_face_images.append(img_name)
            continue

        # choose largest face (dominant person)
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        box = boxes[np.argmax(areas)]

#        x1, y1, x2, y2 = (box / scale).astype(int)
 #       face = image[y1:y2, x1:x2]
#
 #       if face.size == 0:
  #          no_face_images.append(img_name)
   #         continue
        x1, y1, x2, y2 = (box / scale).astype(int)

# 🔴 FIX 2: skip very small faces (low quality)
        if (x2 - x1) < 60 or (y2 - y1) < 60:
            no_face_images.append(img_name)
            continue

        face = image[y1:y2, x1:x2]
        if face.size == 0:
            no_face_images.append(img_name)
            continue


        face = cv2.resize(face, (160, 160))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        face_tensor = np.transpose(face, (2, 0, 1)) / 255.0
        face_tensor = torch.tensor(face_tensor).unsqueeze(0).float()

        embedding = embedder(face_tensor).numpy()[0]
        embedding = embedding / np.linalg.norm(embedding)

        embeddings.append(embedding)
        image_names.append(img_name)

    # -----------------------------
    # Cluster identities
    # -----------------------------
    #labels = DBSCAN(
    #    eps=0.85,
     #   min_samples=1
    #).fit(embeddings).labels_
    labels = DBSCAN(
    eps=0.35,               # cosine threshold
    min_samples=1,
    metric="cosine"
).fit(embeddings).labels_


    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Copy face-detected images
    for img, label in zip(image_names, labels):
        person_dir = os.path.join(output_dir, f"person_{label}")
        os.makedirs(person_dir, exist_ok=True)

        shutil.copy(
            os.path.join(extract_dir, img),
            os.path.join(person_dir, img)
        )

    # Copy no-face images
    if no_face_images:
        unknown_dir = os.path.join(output_dir, "unknown")
        os.makedirs(unknown_dir, exist_ok=True)
        for img in no_face_images:
            shutil.copy(
                os.path.join(extract_dir, img),
                os.path.join(unknown_dir, img)
            )

    # Zip result
    result_zip = os.path.join(base_dir, "result.zip")
    shutil.make_archive(result_zip.replace(".zip", ""), "zip", output_dir)

    return FileResponse(result_zip, filename="separated_photos.zip")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", reload=True)
