#!/usr/bin/env python3
"""
Database fix script for mimic_runs table.

This script handles the schema mismatch by:
1. Checking if mimic_runs table exists
2. If it exists but is missing columns, it adds them
3. If it doesn't exist, it creates the full table with all columns

Usage:
    python fix_database.py
"""

import sys
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import models to ensure they're registered
from app.db.base import Base
from app.models import *
from app.models.mimic_run import MIMICRun

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ ERROR: DATABASE_URL not found in .env file")
    sys.exit(1)

print(f"🔗 Connecting to database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else DATABASE_URL}")

engine = create_engine(DATABASE_URL)
inspector = inspect(engine)

def check_table_exists(table_name):
    """Check if a table exists in the database."""
    return table_name in inspector.get_table_names()

def get_table_columns(table_name):
    """Get list of column names for a table."""
    if not check_table_exists(table_name):
        return []
    return [col['name'] for col in inspector.get_columns(table_name)]

def fix_mimic_runs_table():
    """Fix the mimic_runs table schema."""
    
    print("\n📊 Checking mimic_runs table...")
    
    if not check_table_exists('mimic_runs'):
        print("⚠️  Table 'mimic_runs' does not exist. Creating it...")
        MIMICRun.__table__.create(engine)
        print("✅ Table 'mimic_runs' created successfully with all columns")
        return
    
    print("✓ Table 'mimic_runs' exists")
    
    # Get existing columns
    existing_columns = get_table_columns('mimic_runs')
    print(f"📋 Existing columns: {', '.join(existing_columns)}")
    
    # Define expected columns from the model
    expected_columns = {
        'id': 'INTEGER PRIMARY KEY',
        'run_id': 'VARCHAR UNIQUE NOT NULL',
        'dataset_name': 'VARCHAR',
        'result_path': 'VARCHAR',
        'status': 'VARCHAR DEFAULT \'processing\'',
        'parameters': 'JSON',
        'metrics': 'JSON',
        'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
        'completed_at': 'TIMESTAMP',
        'error_message': 'VARCHAR',
        'user_id': 'INTEGER REFERENCES users(id)'
    }
    
    # Find missing columns
    missing_columns = [col for col in expected_columns.keys() if col not in existing_columns]
    
    if not missing_columns:
        print("✅ All columns are present. No migration needed.")
        return
    
    print(f"\n⚠️  Missing columns detected: {', '.join(missing_columns)}")
    print("🔧 Adding missing columns...")
    
    # Add missing columns
    with engine.connect() as conn:
        for col_name in missing_columns:
            col_def = expected_columns[col_name]
            
            # Simplify column definition for ALTER TABLE
            if col_name == 'run_id':
                sql = f"ALTER TABLE mimic_runs ADD COLUMN {col_name} VARCHAR UNIQUE"
            elif col_name == 'status':
                sql = f"ALTER TABLE mimic_runs ADD COLUMN {col_name} VARCHAR DEFAULT 'processing'"
            elif col_name in ['parameters', 'metrics']:
                sql = f"ALTER TABLE mimic_runs ADD COLUMN {col_name} JSON"
            elif col_name in ['created_at', 'completed_at']:
                sql = f"ALTER TABLE mimic_runs ADD COLUMN {col_name} TIMESTAMP"
            elif col_name == 'user_id':
                sql = f"ALTER TABLE mimic_runs ADD COLUMN {col_name} INTEGER REFERENCES users(id)"
            else:
                sql = f"ALTER TABLE mimic_runs ADD COLUMN {col_name} VARCHAR"
            
            try:
                conn.execute(text(sql))
                conn.commit()
                print(f"  ✓ Added column: {col_name}")
            except Exception as e:
                print(f"  ✗ Failed to add column {col_name}: {e}")
                conn.rollback()
    
    print("\n✅ Database schema fix completed!")

def verify_all_tables():
    """Verify all expected tables exist."""
    print("\n🔍 Verifying all tables...")
    
    expected_tables = ['users', 'roles', 'course_materials', 'knowledge_chunks', 'mimic_runs']
    existing_tables = inspector.get_table_names()
    
    for table in expected_tables:
        if table in existing_tables:
            columns = get_table_columns(table)
            print(f"  ✓ {table} ({len(columns)} columns)")
        else:
            print(f"  ✗ {table} (missing)")
    
    print()

if __name__ == "__main__":
    try:
        print("=" * 60)
        print("🔧 MIMIC Platform Database Fix Script")
        print("=" * 60)
        
        verify_all_tables()
        fix_mimic_runs_table()
        verify_all_tables()
        
        print("\n" + "=" * 60)
        print("✅ Database fix completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
