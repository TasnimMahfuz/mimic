from app.db.session import SessionLocal
from app.models.role import Role
 
db = SessionLocal()
roles = db.query(Role).all()
print('Existing roles in database:')
for role in roles:
    print(f'  - ID: {role.id}, Name: {role.name}')
db.close()
