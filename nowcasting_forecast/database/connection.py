""" Database Connection class"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session


class DatabaseConnection:
    """Database connection class"""

    def __init__(self, url):
        """
        Set up database connection

        url: the database url, used for connecting
        """
        self.url = url

        self.engine = create_engine(self.url, echo=True)
        self.Session = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        """Get sqlalamcy session"""
        return self.Session()
