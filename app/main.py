from fastapi import FastAPI
from datetime import datetime

app = FastAPI()

@app.get("/")
def read_root():
    current_time = datetime.utcnow().isoformat()
    return {"message": f"Hello, ! Current time: {current_time}"}
