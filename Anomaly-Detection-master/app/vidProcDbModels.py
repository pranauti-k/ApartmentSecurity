from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy import create_engine

from sqlalchemy.orm import declarative_base
#from sqlalchemy.ext.declarative import declarative_base
import datetime

dbName = 'vidlogs.db'
#engine = create_engine('sqlite:///vidlogs.db', echo = True)
#** IMP: had to set same_thread to false otherwise connection to sql db is blocked by
#watchdog thread!!
#If you want to see the sql stuff echoed on the commandline, then
#change echo = True from False in the sqllite connection.
#engine = create_engine("sqlite:///" + dbName + "?check_same_thread=false" , echo = True)
engine = create_engine("sqlite:///" + dbName + "?check_same_thread=false" , echo = False)
Base = declarative_base()

class User(Base):
    #__bind_key__ = 'udb'
    __tablename__ = 'users_table'
    id = Column(Integer, primary_key=True)
    name = Column(String(15))
    username = Column(String(15), unique=True)
    email = Column(String(50), unique=True)
    #No need for unique= True
    password = Column(String(256), unique=False)
 
class VidLogs(Base):
    __tablename__ = 'vidlogs_table'
    sno = Column(Integer, primary_key = True)
    fname = Column(String(200), nullable = False)
    label = Column(String(500), nullable = False)
    date_created = Column(DateTime, default = datetime.datetime.now)
 
    def __repr__(self) -> str:
        return f"{self.sno} - {self.fname}"

Base.metadata.create_all(engine)

from sqlalchemy.orm import sessionmaker
Session = sessionmaker(bind = engine)
dbSession = Session()

