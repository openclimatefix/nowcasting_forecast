from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session


class DatabaseConnection:

    def __init__(self, url):
        self.url = url

        self.engine = create_engine(self.url, echo=True)
        self.Session = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:

        return self.Session()
