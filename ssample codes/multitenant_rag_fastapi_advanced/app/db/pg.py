
from sqlalchemy import create_engine
import os

engine = create_engine(os.getenv("PG_URL"))

def get_engine():
    return engine
