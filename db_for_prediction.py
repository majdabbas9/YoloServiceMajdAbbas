from abc import ABC, abstractmethod
import sqlite3
import boto3
from datetime import datetime
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key
DB_PATH = "predictions.db"
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
# === Abstract Base Class ===
class BaseDatabaseHandler(ABC):
    @abstractmethod
    def init_db(self):
        pass

    @abstractmethod
    def save_prediction_session(self, uid, original_image, predicted_image):
        pass

    @abstractmethod
    def save_detection_object(self,c,prediction_uid, label, score, box):
        pass
    @abstractmethod
    def get_predicted_image(self, uid):
        pass
    @abstractmethod
    def get_prediction_by_uid(self,uid):
        pass
    @abstractmethod
    def get_predictions_by_label(self,label: str):
        pass
    @abstractmethod
    def get_predictions_by_score(self,min_score: float):
        pass
    @abstractmethod
    def get_prediction_image(self,uid: str, request: Request):
        pass


# === SQLite Implementation ===
class SQLiteDatabaseHandler(BaseDatabaseHandler):
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_sessions (
                    uid TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    original_image TEXT,
                    predicted_image TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detection_objects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_uid TEXT,
                    label TEXT,
                    score REAL,
                    box TEXT,
                    FOREIGN KEY (prediction_uid) REFERENCES prediction_sessions (uid)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_uid ON detection_objects (prediction_uid)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON detection_objects (label)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_score ON detection_objects (score)")

    def save_prediction_session(self,uid, original_image, predicted_image):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO prediction_sessions (uid, original_image, predicted_image)
                VALUES (?, ?, ?)
            """, (uid, original_image, predicted_image))

    def save_detection_object(self,c,prediction_uid, label, score, box):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO detection_objects (prediction_uid, label, score, box)
                VALUES (?, ?, ?, ?)
            """, (prediction_uid, label, score, str(box)))

    def get_predicted_image(self, uid):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT predicted_image FROM prediction_sessions WHERE uid = ?
            """, (uid,))
            row = cursor.fetchone()
            return row[0] if row else None

    def get_prediction_by_uid(self,uid: str):
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

    def get_predictions_by_label(self,label: str):
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
            return [{"uid": row["uid"]} for row in rows]

    def get_predictions_by_score(self,min_score: float):
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
            return [{"uid": row["uid"]} for row in rows]

    def get_prediction_image(self,uid: str, request: Request):
        """
        Get prediction image by uid
        """
        import os
        from fastapi.responses import FileResponse
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

# === DynamoDB Implementation ===
class DynamoDBDatabaseHandler(BaseDatabaseHandler):
    def __init__(self, env='dev', project_prefix='majd_yolo'):
        self.dynamodb = boto3.resource('dynamodb', region_name='eu-west-1')
        # Compose full prefix using environment + project prefix
        self.prefix = f"{project_prefix}_{env}"  # e.g. "dev_majd_yolo" or "prod_majd_yolo"
        self.prediction_sessions_table = self.dynamodb.Table(f"{self.prefix}_prediction_session")
        self.detection_objects_table = self.dynamodb.Table(f"{self.prefix}_detection_objects")

    def init_db(self):
        # Optional: Validate tables exist
        try:
            self.prediction_sessions_table.load()
            self.detection_objects_table.load()
        except ClientError as e:
            raise RuntimeError("One or more DynamoDB tables do not exist") from e

    def save_prediction_session(self, uid, original_image, predicted_image):
        self.prediction_sessions_table.put_item(Item={
            'uid': uid,
            'timestamp': datetime.utcnow().isoformat(),
            'original_image': original_image,
            'predicted_image': predicted_image
        })

    def save_detection_object(self,c,prediction_uid, label, score, box):
        from decimal import Decimal
        score = Decimal(score)
        box = [Decimal(coord) for coord in box]
        self.detection_objects_table.put_item(Item={
            'prediction_uid': prediction_uid,
            'score': f'{score}_{c}',
            'label': label,
            'score_partition' : 'score',
            'label_score': score,
            'box': str(box)
        })

    def get_predicted_image(self, uid):
        response = self.prediction_sessions_table.get_item(Key={'uid': uid})
        item = response.get('Item')
        return item.get('predicted_image') if item else None

    def get_prediction_by_uid(self, uid: str):
        """
        Get prediction session by uid with all detected objects
        """
        # 1. Fetch prediction session
        try:
            response = self.prediction_sessions_table.get_item(Key={'uid': uid})
            session = response.get('Item')
            if not session:
                raise HTTPException(status_code=404, detail="Prediction not found")
        except ClientError as e:
            raise HTTPException(status_code=500, detail="Failed to fetch prediction session") from e

        # 2. Fetch detection objects
        try:
            response = self.detection_objects_table.query(
                KeyConditionExpression=Key('prediction_uid').eq(uid)
            )
            objects = response.get('Items', [])
        except ClientError as e:
            raise HTTPException(status_code=500, detail="Failed to fetch detection objects") from e

        # 3. Format and return result
        return {
            "uid": session["uid"],
            "timestamp": session.get("timestamp"),
            "original_image": session.get("original_image"),
            "predicted_image": session.get("predicted_image"),
            "detection_objects": [
                {
                    "label": obj.get("label"),
                    "score": obj.get("label_score", 0),
                    "box": obj.get("box", [])
                } for obj in objects
            ]
        }

    def get_predictions_by_label(self,label: str):
        response = self.detection_objects_table.query(
            IndexName='label-index',
            KeyConditionExpression=Key('label').eq(label)
        )
        return response['Items']

    def get_predictions_by_score(self,min_score: float):
        from decimal import Decimal
        min_score = Decimal(min_score)
        response = self.detection_objects_table.query(
            IndexName='score_partition-score-index',
            KeyConditionExpression=Key('score_partition').eq('score') & Key('label_score').gte(min_score)
        )
        return response['Items']

    def get_prediction_image(self, uid: str, request: Request):
        """
        Get prediction image by uid
        """
        import os
        from fastapi.responses import FileResponse
        accept = request.headers.get("accept", "")
        try:
            response = self.prediction_sessions_table.get_item(Key={'uid': uid})
            item = response.get('Item')
            if not item:
                raise HTTPException(status_code=404, detail="Prediction not found")
            image_path = item['predicted_image']
        except ClientError as e:
            raise HTTPException(status_code=500, detail="Failed to fetch prediction image") from e

        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Predicted image file not found")

        if "image/png" in accept:
            return FileResponse(image_path, media_type="image/png")
        elif "image/jpeg" in accept or "image/jpg" in accept:
            return FileResponse(image_path, media_type="image/jpeg")
        else:
            # If the client doesn't accept image, respond with 406 Not Acceptable
            raise HTTPException(status_code=406, detail="Client does not accept an image format")

# === Factory Method ===
class DatabaseFactory:
    @staticmethod
    def create_database(db_type, **kwargs) -> BaseDatabaseHandler:
        if db_type == 'sqlite':
            return SQLiteDatabaseHandler(kwargs['db_path'])
        elif db_type == 'dynamodb':
            env = kwargs.get('env', 'dev')  # default to dev
            project_prefix = kwargs.get('table_prefix', 'majd_yolo')
            return DynamoDBDatabaseHandler(env=env, project_prefix=project_prefix)
        else:
            raise ValueError("Unsupported db_type. Use 'sqlite' or 'dynamodb'.")