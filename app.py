from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from decimal import Decimal
from fastapi.responses import FileResponse, Response
from ultralytics import YOLO
from PIL import Image
import sqlite3
import threading
import os
import uuid
import shutil
import boto3
from S3_requests import upload_file,download_file
import requests
# Disable GPU usage
import torch
import time
from db_for_prediction import DatabaseFactory
import json
torch.cuda.is_available = lambda: False
app = FastAPI()
S3_bucket_name = os.getenv('S3_BUCKET_NAME')
storage_type = os.getenv("STORAGE_TYPE", "sqlite")
Queue_URL = os.getenv("QUEUE_URL")
Polybot_url = os.getenv("POLYBOT_URL")
UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"
DB_PATH = "predictions.db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)
sqs = boto3.client('sqs', region_name='eu-west-1')
# Download the AI model (tiny model ~6MB)
model = YOLO("yolov8n.pt")  

ENVIRONMENT = 'dev' if 'dev' in S3_bucket_name.lower() else 'prod'
# Initialize database
if storage_type == "sqlite":
    # make sure DB_PATH is defined or imported from your config
    db = DatabaseFactory.create_database("sqlite", db_path=DB_PATH)
elif storage_type == "dynamodb":
    # you can optionally pass a custom prefix for your Dynamo tables
    db = DatabaseFactory.create_database(
        "dynamodb",
        env=ENVIRONMENT,
        table_prefix='majd_yolo'
    )
else:
    raise ValueError(f"Unknown STORAGE_TYPE {storage_type!r}")


def poll_sqs_messages():
    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=Queue_URL,
                MaxNumberOfMessages=5,
                WaitTimeSeconds=20
            )
            messages = response.get('Messages', [])

            for msg in messages:
                msg_body = json.loads(msg['Body'])
                s3_key = msg_body['s3_key']
                chat_id = msg_body['chat_id']
                file_path = msg_body['file_path']
                uid = str(uuid.uuid4())
                ext = '.' + s3_key.split('.')[-1]
                original_path = os.path.join(UPLOAD_DIR, uid + ext)
                download_file(S3_bucket_name, s3_key, original_path)
                predicted_path = os.path.join(PREDICTED_DIR, uid + ext)
                results = model(original_path, device="cpu")
                annotated_frame = results[0].plot()  # NumPy image with boxes
                annotated_image = Image.fromarray(annotated_frame)
                annotated_image.save(predicted_path)

                db.save_prediction_session(uid, original_path, predicted_path)
                detected_labels = []

                for box in results[0].boxes:
                    label_idx = int(box.cls[0].item())
                    label = model.names[label_idx]
                    score = Decimal(box.conf[0].item())
                    bbox_raw = box.xyxy[0].tolist()
                    bbox = [Decimal(x) for x in bbox_raw]
                    db.save_detection_object(uid, label, score, bbox)
                    detected_labels.append(label)
                upload_file(predicted_path, S3_bucket_name, f'yolo_to_poly_images/{s3_key.split("/")[-1]}')
                time.sleep(1.5)
                # send a post request to Polybot with uid chat_id file_path as json
                # Notify Polybot
                payload = {
                    "uid": uid,
                    "chat_id": chat_id,
                    "file_path": file_path,
                    "image_url": f'yolo_to_poly_images/{s3_key.split("/")[-1]}'
                }

                try:
                    polybot_response = requests.post(f'{Polybot_url}/predictions', json=payload)
                    polybot_response.raise_for_status()
                except Exception as post_err:
                    print(f"[Polybot Callback Error] {post_err}")
                sqs.delete_message(QueueUrl=Queue_URL, ReceiptHandle=msg['ReceiptHandle'])
            if not messages:
                time.sleep(1)

        except Exception as e:
            print(f"[SQS Polling Error] {e}")
            time.sleep(5)  # avoid tight retry loop

@app.on_event("startup")
def start_sqs_polling():
    print("Starting SQS polling thread...")
    thread = threading.Thread(target=poll_sqs_messages, daemon=True)
    thread.start()

@app.post("/predict")
def predict(s3_key:str):
    """
    Predict objects in an image
    """

    # ext = os.path.splitext(file.filename)[1]
    uid = str(uuid.uuid4())
    ext = '.'+s3_key.split('.')[-1]
    original_path = os.path.join(UPLOAD_DIR, uid + ext)
    download_file(S3_bucket_name,s3_key,original_path)
    predicted_path = os.path.join(PREDICTED_DIR, uid + ext)
    results = model(original_path, device="cpu")
    annotated_frame = results[0].plot()  # NumPy image with boxes
    annotated_image = Image.fromarray(annotated_frame)
    annotated_image.save(predicted_path)

    db.save_prediction_session(uid, original_path, predicted_path)
    detected_labels = []

    for box in results[0].boxes:
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        score = Decimal(box.conf[0].item())
        bbox_raw = box.xyxy[0].tolist()
        bbox = [Decimal(x) for x in bbox_raw]
        db.save_detection_object(uid, label, score, bbox)
        detected_labels.append(label)
    upload_file(predicted_path,S3_bucket_name, f'yolo_to_poly_images/{s3_key.split("/")[-1]}')
    return {
        "prediction_uid": uid,
        "detection_count": len(results[0].boxes),
        "labels": detected_labels
    }

@app.get("/prediction/{uid}")
def get_prediction_by_uid(uid: str):
    """
    Get prediction session by uid with all detected objects
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        # Get prediction session
        session = conn.execute("SELECT * FROM prediction_sessions WHERE uid = ?", (uid,)).fetchone()
        if not session:
            raise HTTPException(status_code=404, detail="Prediction not found")
            
        # Get all detection objects for this prediction
        objects = conn.execute(
            "SELECT * FROM detection_objects WHERE prediction_uid = ?", 
            (uid,)
        ).fetchall()
        
        return {
            "uid": session["uid"],
            "timestamp": session["timestamp"],
            "original_image": session["original_image"],
            "predicted_image": session["predicted_image"],
            "detection_objects": [
                {
                    "id": obj["id"],
                    "label": obj["label"],
                    "score": obj["score"],
                    "box": obj["box"]
                } for obj in objects
            ]
        }

@app.get("/predictions/label/{label}")
def get_predictions_by_label(label: str):
    """
    Get prediction sessions containing objects with specified label
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT DISTINCT ps.uid, ps.timestamp
            FROM prediction_sessions ps
            JOIN detection_objects do ON ps.uid = do.prediction_uid
            WHERE do.label = ?
        """, (label,)).fetchall()
        return [{"uid": row["uid"], "timestamp": row["timestamp"]} for row in rows]

@app.get("/predictions/score/{min_score}")
def get_predictions_by_score(min_score: float):
    """
    Get prediction sessions containing objects with score >= min_score
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT DISTINCT ps.uid, ps.timestamp
            FROM prediction_sessions ps
            JOIN detection_objects do ON ps.uid = do.prediction_uid
            WHERE do.score >= ?
        """, (min_score,)).fetchall()
        return [{"uid": row["uid"], "timestamp": row["timestamp"]} for row in rows]

@app.get("/image/{type}/{filename}")
def get_image(type: str, filename: str):
    """
    Get image by type and filename
    """
    if type not in ["original", "predicted"]:
        raise HTTPException(status_code=400, detail="Invalid image type")
    path = os.path.join("uploads", type, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)

@app.get("/prediction/{uid}/image")
def get_prediction_image(uid: str, request: Request):
    """
    Get prediction image by uid
    """
    accept = request.headers.get("accept", "")
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT predicted_image FROM prediction_sessions WHERE uid = ?", (uid,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Prediction not found")
        image_path = row[0]

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Predicted image file not found")

    if "image/png" in accept:
        return FileResponse(image_path, media_type="image/png")
    elif "image/jpeg" in accept or "image/jpg" in accept:
        return FileResponse(image_path, media_type="image/jpeg")
    else:
        # If the client doesn't accept image, respond with 406 Not Acceptable
        raise HTTPException(status_code=406, detail="Client does not accept an image format")

@app.get("/health")
def health():
    """
    Health check endpoint
    """
    return {"status": "ok","message": "Service is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
