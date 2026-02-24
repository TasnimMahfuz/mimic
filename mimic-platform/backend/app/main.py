import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import auth, mimic, materials, chat
from app.db.base import Base
from app.db.session import engine, SessionLocal
from app.db.init_db import init_db


# Initialize database tables
Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI(title="MIMIC Platform API")

# ========== CORS MIDDLEWARE (MUST BE BEFORE ROUTERS) ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",      # Vite dev server
        "http://localhost:3000",       # Alternative dev port
        "http://127.0.0.1:5173",       # Localhost variant
        "http://127.0.0.1:3000",       # Localhost variant
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ========== ROUTERS (AFTER MIDDLEWARE) ==========
app.include_router(auth.router, prefix="/auth")
app.include_router(mimic.router, prefix="/mimic")
app.include_router(materials.router, prefix="/materials")
app.include_router(chat.router, prefix="/chat")

# ========== HEALTH CHECK ==========
@app.get("/")
def root():
    return {"message": "MIMIC backend running"}


# ========== STARTUP EVENT ==========
@app.on_event("startup")
def startup():
    """Initialize database on application startup."""
    if os.getenv("TESTING"):
        print("Skipping startup DB init (TEST MODE)")
        return

    db = SessionLocal()
    try:
        init_db(db)
        print("✓ Database initialized successfully")
    except Exception as e:
        print(f"✗ Database initialization error: {e}")
    finally:
        db.close()