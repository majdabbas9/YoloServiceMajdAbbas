from abc import ABC, abstractmethod
import sqlite3
import boto3
from datetime import datetime
from botocore.exceptions import ClientError

# === Abstract Base Class ===
class BaseDatabaseHandler(ABC):
    @abstractmethod
    def init_db(self):
        pass

    @abstractmethod
    def save_prediction_session(self, uid, original_image, predicted_image):
        pass

    @abstractmethod
    def save_detection_object(self, prediction_uid, label, score, box):
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

    def save_prediction_session(self, uid, original_image, predicted_image):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO prediction_sessions (uid, original_image, predicted_image)
                VALUES (?, ?, ?)
            """, (uid, original_image, predicted_image))

    def save_detection_object(self, prediction_uid, label, score, box):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO detection_objects (prediction_uid, label, score, box)
                VALUES (?, ?, ?, ?)
            """, (prediction_uid, label, score, str(box)))


# === DynamoDB Implementation ===
class DynamoDBDatabaseHandler(BaseDatabaseHandler):
    def __init__(self, table_prefix='majd_yolo'):
        self.dynamodb = boto3.resource('dynamodb',region_name='eu-west-1')
        self.prefix = table_prefix
        self.prediction_sessions_table = self.dynamodb.Table(f"{self.prefix}_prediction_sessions")
        self.detection_objects_table = self.dynamodb.Table(f"{self.prefix}_detection_objects")
        # self.init_db()  # <-- No need to call this anymore if tables already exist

    def init_db(self):
        # Optional: Just validate the tables exist
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

    def save_detection_object(self, prediction_uid, label, score, box):
        self.detection_objects_table.put_item(Item={
            'id': f"{prediction_uid}_{label}_{datetime.utcnow().timestamp()}",
            'prediction_uid': prediction_uid,
            'label': label,
            'score': float(score),
            'box': str(box)
        })


# === Factory Method ===
class DatabaseFactory:
    @staticmethod
    def create_database(db_type, **kwargs) -> BaseDatabaseHandler:
        if db_type == 'sqlite':
            return SQLiteDatabaseHandler(kwargs['db_path'])
        elif db_type == 'dynamodb':
            return DynamoDBDatabaseHandler(kwargs.get('table_prefix', 'ml'))
        else:
            raise ValueError("Unsupported db_type. Use 'sqlite' or 'dynamodb'.")