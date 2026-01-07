# 📸 Photo_separator

✨ **Photo Separator using Face Recognition & Unsupervised Clustering** ✨

Photo_separator is a Machine Learning–based web application that automatically **groups photos of the same person into a single folder**, even when photos are taken in groups.  
The user uploads a ZIP file of images, and the system intelligently organizes them person-wise without any prior labels.

---

## 🚀 Features

✅ Upload a ZIP file containing photos  
✅ Detects faces automatically  
✅ Groups **same person’s photos into one folder**  
✅ Handles **group photos** correctly  
✅ Keeps all photos (no image is dropped)  
✅ Places unclear photos into an `unknown` folder  
✅ Simple and clean web interface  
✅ Fast CPU-based processing  

---

## 🧠 Machine Learning Concepts Used

This project combines multiple **Machine Learning and Deep Learning concepts**:

### 🔹 Unsupervised Learning (Core Concept)
- The system does **not use labeled data**
- It automatically discovers patterns in face embeddings
- Person identities are formed using **clustering**, not classification

### 🔹 Convolutional Neural Networks (CNN)
- CNNs are used internally in:
  - **MTCNN** for face detection
  - **FaceNet (InceptionResnetV1)** for feature extraction
- CNNs learn spatial facial features such as eyes, nose, and face structure

### 🔹 Face Detection – MTCNN
- **MTCNN (Multi-task Cascaded Convolutional Neural Network)**
- Detects faces in images
- Works well for single-person and group photos

### 🔹 Face Feature Extraction – FaceNet
- Uses **InceptionResnetV1 (FaceNet architecture)**
- Converts each face into a **128-dimensional embedding**
- Similar faces produce similar numerical representations

### 🔹 Clustering Algorithm – DBSCAN
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
- Groups similar face embeddings automatically
- Does not require the number of people in advance
- Identifies and handles outliers naturally

---

## ⚙️ How the Project Works

1️⃣ User uploads a ZIP file containing photos  
2️⃣ ZIP file is extracted on the server  
3️⃣ Each image is processed:
   - Face detection using **MTCNN**
   - The **largest (dominant) face** is selected per image  
4️⃣ Face embeddings are generated using **FaceNet**  
5️⃣ Embeddings are **normalized** for accurate similarity comparison  
6️⃣ **DBSCAN clustering** groups faces into person identities  
7️⃣ Output folders are created automatically:
   - `person_0`, `person_1`, `person_2`, etc.  
8️⃣ Photos are copied into corresponding person folders  
9️⃣ Photos where:
   - faces are not detected
   - faces are too small or unclear  
   are placed into a separate folder:
   - 📁 `unknown/`  
🔟 Final output is compressed into a ZIP file and downloaded  


---

## 🖥️ Technologies Used

- Python  
- FastAPI  
- PyTorch  
- facenet-pytorch  
- OpenCV  
- scikit-learn  
- HTML  
- CSS  

---

## ⚠️ Drawbacks & Limitations

⚠️ Since this project is based on **Unsupervised Learning**, some limitations exist:

- 👀 **Side-facing, blurry, or low-light photos** may not detect faces correctly and are placed in the `unknown` folder  
- 😕 **Photos of the same person may sometimes be classified into different folders** if:
  - Lighting conditions vary significantly  
  - Face angles change a lot  
  - Faces are partially covered (mask, glasses, cap, etc.)  
- 🧠 DBSCAN clustering may:
  - Split one person into multiple folders (over-segmentation)
  - Occasionally merge very similar-looking people  
- 🚫 The system does **not perform identity recognition** (no names or labels)
- ⚡ Performance depends on:
  - Number of photos
  - Image resolution
  - CPU capability  

These drawbacks are **expected and acceptable** for an unsupervised face-clustering system.

---

## 🎓 Academic Relevance

This project demonstrates:
- Unsupervised Machine Learning
- CNN-based feature extraction
- Real-world limitations of ML systems

It is suitable for:
- Machine Learning projects  
- Computer Vision coursework  
- Mini / Major academic projects  

---

## 🔮 Future Improvements

- Use hierarchical clustering for better grouping  
- Add manual merge / rename options  
- Improve face quality scoring  
- GPU acceleration  
- Progress indicator in UI  

---

## 🙌 Conclusion

**Photo_separator** shows how deep learning and unsupervised clustering can be applied to organize personal photo collections automatically, while also highlighting the real-world challenges of face-based grouping.



