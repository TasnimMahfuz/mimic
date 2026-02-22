from fastapi import FastAPI
from app.api.routes import auth, mimic, materials, chat
from app.db.base import Base
from app.db.session import engine
from app.db.init_db import init_db
import os


Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(auth.router, prefix="/auth")
app.include_router(mimic.router, prefix="/mimic")
app.include_router(materials.router, prefix="/materials")
app.include_router(chat.router, prefix="/chat")

#@app.on_event("startup")
#def on_startup():
#    init_db()


import os
from app.db.session import SessionLocal
from app.db.init_db import init_db


@app.on_event("startup")
def startup():

    if os.getenv("TESTING"):
        print("Skipping startup DB init (TEST MODE)")
        return

    db = SessionLocal()
    try:
        init_db(db)
    finally:
        db.close()
        
@app.get("/")
def root():
    return {"message": "MIMIC backend running"}