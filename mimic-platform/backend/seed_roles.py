#!/usr/bin/env python3
"""
Seed script to populate default roles in the database.
Run this script to create 'teacher' and 'student' roles if they don't exist.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.db.session import SessionLocal
from app.models.role import Role

def seed_roles():
    """Create default roles if they don't exist."""
    db = SessionLocal()
    
    default_roles = ["teacher", "student"]
    created_roles = []
    existing_roles = []
    
    try:
        for role_name in default_roles:
            existing_role = db.query(Role).filter(Role.name == role_name).first()
            
            if existing_role:
                existing_roles.append(role_name)
                print(f"Role '{role_name}' already exists")
            else:
                new_role = Role(name=role_name)
                db.add(new_role)
                created_roles.append(role_name)
                print(f"Created role: {role_name}")
        
        db.commit()
        
        print(f"\nSeeding complete:")
        print(f"  - Created: {created_roles}")
        print(f"  - Already existed: {existing_roles}")
        
    except Exception as e:
        db.rollback()
        print(f"Error seeding roles: {e}")
        return False
    finally:
        db.close()
    
    return True

if __name__ == "__main__":
    print("Starting role seeding...")
    success = seed_roles()
    if success:
        print("✅ Role seeding completed successfully")
    else:
        print("❌ Role seeding failed")
        sys.exit(1)
