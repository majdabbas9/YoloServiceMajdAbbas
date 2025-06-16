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
    def __init__(self, table_prefix='ml'):
        self.dynamodb = boto3.resource('dynamodb')
        self.client = boto3.client('dynamodb')
        self.prefix = table_prefix
        self.init_db()

    def init_db(self):
        self._create_table_if_not_exists(
            f"{self.prefix}_prediction_sessions",
            [{'AttributeName': 'uid', 'KeyType': 'HASH'}],
            [{'AttributeName': 'uid', 'AttributeType': 'S'}]
        )

        self._create_table_if_not_exists(
            f"{self.prefix}_detection_objects",
            [{'AttributeName': 'id', 'KeyType': 'HASH'}],
            [{'AttributeName': 'id', 'AttributeType': 'S'}]
        )

    def _create_table_if_not_exists(self, table_name, key_schema, attr_definitions):
        try:
            self.client.describe_table(TableName=table_name)
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                self.dynamodb.create_table(
                    TableName=table_name,
                    KeySchema=key_schema,
                    AttributeDefinitions=attr_definitions,
                    BillingMode='PAY_PER_REQUEST'
                )
                print(f"Creating table {table_name}...")
                waiter = self.client.get_waiter('table_exists')
                waiter.wait(TableName=table_name)
            else:
                raise

    def save_prediction_session(self, uid, original_image, predicted_image):
        table = self.dynamodb.Table(f"{self.prefix}_prediction_sessions")
        table.put_item(Item={
            'uid': uid,
            'timestamp': datetime.utcnow().isoformat(),
            'original_image': original_image,
            'predicted_image': predicted_image
        })

    def save_detection_object(self, prediction_uid, label, score, box):
        table = self.dynamodb.Table(f"{self.prefix}_detection_objects")
        table.put_item(Item={
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