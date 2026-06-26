import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_NAME = 'logistics_logs.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (timestamp TEXT, ship_lat REAL, ship_lon REAL,
                  port_lat REAL, port_lon REAL, news_text TEXT, predicted_delay REAL)''')
    conn.commit()
    conn.close()

def log_prediction(ship_lat, ship_lon, port_lat, port_lon, news_text, delay):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?)",
              (now, ship_lat, ship_lon, port_lat, port_lon, news_text, delay))
    conn.commit()
    conn.close()

def get_all_logs():
    if not os.path.exists(DB_NAME):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()
    return df