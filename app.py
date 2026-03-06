import os
import zipfile
import shutil
import uuid

import cv2
import numpy as np
import torch
from PIL import Image

from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.cluster import DBSCAN
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

torch.set_grad_enabled(False)

device = "cpu"
detector = MTCNN(keep_all=True, device=device, post_process=False)
embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def resize_for_detection(image, max_size=800):
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image, scale

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

    for img_name in os.listdir(extract_dir):
        if not img_name.lower().endswith(IMAGE_EXTS):
            continue

        img_path = os.path.join(extract_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        try:
            small, scale = resize_for_detection(image)
            boxes, probs = detector.detect(small)
            
            if boxes is None or len(boxes) == 0:
                no_face_images.append(img_name)
                continue

            # Pick largest face (by area)
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            best_idx = np.argmax(areas)
            box = boxes[best_idx]
            
            x1, y1, x2, y2 = (box / scale).astype(int)
            
            # Add margin
            margin = 20
            h, w = image.shape[:2]
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            
            if (x2 - x1) < 80 or (y2 - y1) < 80:
                no_face_images.append(img_name)
                continue

            face = image[y1:y2, x1:x2]
            if face.size == 0:
                no_face_images.append(img_name)
                continue

            face = cv2.resize(face, (160, 160))
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
            face_tensor = np.transpose(face_rgb, (2, 0, 1)) / 255.0
            face_tensor = torch.tensor(face_tensor).unsqueeze(0).float().to(device)

            with torch.no_grad():
                embedding = embedder(face_tensor).cpu().numpy()[0]
            
            # L2 normalize
            embedding = embedding / np.linalg.norm(embedding)

            embeddings.append(embedding)
            image_names.append(img_name)
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            no_face_images.append(img_name)
            continue

    # 🔴 FIXED CLUSTERING - Using cosine distance which is better for faces
    if embeddings:
        embeddings_array = np.array(embeddings)
        
        # DBSCAN with cosine distance (better for face embeddings)
        clustering = DBSCAN(
            eps=0.4,              # Lower threshold for cosine distance
            min_samples=1,
            metric='cosine'       # Use cosine similarity
        )
        labels = clustering.fit_predict(embeddings_array)
        
        print(f"Clustered {len(embeddings)} faces into {len(set(labels))} groups")
        print(f"Labels: {labels}")
    else:
        labels = []

    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    for img, label in zip(image_names, labels):
        person_dir = os.path.join(output_dir, f"person_{label}")
        os.makedirs(person_dir, exist_ok=True)
        shutil.copy(
            os.path.join(extract_dir, img),
            os.path.join(person_dir, img)
        )

    if no_face_images:
        unknown_dir = os.path.join(output_dir, "unknown")
        os.makedirs(unknown_dir, exist_ok=True)
        for img in no_face_images:
            shutil.copy(
                os.path.join(extract_dir, img),
                os.path.join(unknown_dir, img)
            )

    result_zip = os.path.join(base_dir, "result.zip")
    shutil.make_archive(result_zip.replace(".zip", ""), "zip", output_dir)

    return FileResponse(result_zip, filename="separated_photos.zip")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)