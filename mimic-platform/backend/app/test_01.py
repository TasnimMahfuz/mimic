from app.db.session import engine
from app.db.base import Base
from app.models import *
 
print('Tables that will be created:')
for table in Base.metadata.tables.keys():
    print(f'  - {table}')


for table in Base.metadata.tables.keys():
    print(f'Creating table: {table}')
    Base.metadata.tables[table].create(bind=engine, checkfirst=True)