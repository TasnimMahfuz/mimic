#!/usr/bin/env python3
"""
Database reset script for development.

WARNING: This will DROP ALL TABLES and recreate them from scratch.
Only use this in development!

Usage:
    python reset_database.py
"""

import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import models to ensure they're registered
from app.db.base import Base
from app.models import *
from app.models.mimic_run import MIMICRun
from app.db.init_db import init_db

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ ERROR: DATABASE_URL not found in .env file")
    sys.exit(1)

print(f"🔗 Connecting to database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else DATABASE_URL}")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def reset_database():
    """Drop all tables and recreate them."""
    
    print("\n⚠️  WARNING: This will DELETE ALL DATA in the database!")
    response = input("Are you sure you want to continue? (yes/no): ")
    
    if response.lower() != 'yes':
        print("❌ Operation cancelled.")
        sys.exit(0)
    
    print("\n🗑️  Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    print("✅ All tables dropped")
    
    print("\n🔨 Creating all tables from models...")
    Base.metadata.create_all(bind=engine)
    print("✅ All tables created")
    
    print("\n🌱 Seeding initial data...")
    db = SessionLocal()
    try:
        init_db(db)
        print("✅ Initial data seeded")
    except Exception as e:
        print(f"⚠️  Warning during seeding: {e}")
    finally:
        db.close()
    
    print("\n✅ Database reset completed successfully!")

if __name__ == "__main__":
    try:
        print("=" * 60)
        print("🔄 MIMIC Platform Database Reset Script")
        print("=" * 60)
        
        reset_database()
        
        print("\n" + "=" * 60)
        print("✅ Database is ready for use!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
