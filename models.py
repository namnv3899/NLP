from sqlalchemy import Column, DateTime, Integer, String, UnicodeText, create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

from utils import load_yaml


class ArticleDB:
    def __init__(self) -> None:
        sqlite_db_path = load_yaml("paths")["sqlite_db"]
        self.connection_string = f"sqlite:///{sqlite_db_path}"
        self.engine = create_engine(self.connection_string, echo=False)
        self.Session = sessionmaker(bind=self.engine)
        self.create_table()

    def create_table(self):
        try:
            Base.metadata.create_all(bind=self.engine)
            print("Table create successfully!")
        except:
            print("Error while creating table!")

    def create_session(self):
        return self.Session()

    def _insert_obj(self, session, obj):
        try:
            session.add(obj)
            session.commit()
            return obj
        except IntegrityError as e:
            print(e)
            session.rollback()
            return

    def insert_article(self, session, argument):
        article = Article(**argument)
        return self._insert_obj(session, article)

    def get_the_latest_date(self):
        session = self.create_session()
        query = session.query(func.max(Article.datetime).label("newest_date"))
        newest_date = query.first().newest_date
        session.close()
        return newest_date.date() if newest_date is not None else None

    def get_articles_by_specific_categories(self, session, categories=None):
        if categories is None:
            articles = session.query(Article.id, Article.category, Article.body)
        else:
            articles = (
                session.query(Article.id, Article.category, Article.body)
                .filter(Article.category.in_(categories))
                .all()
            )
        return articles

    def get_articles_by_id(self, session, id):
        article = (
            session.query(Article.category, Article.title, Article.body)
            .filter(Article.id == id)
            .first()
        )

        return {
            "category": article[0],
            "title": article[1],
            "body": article[2],
        }


Base = declarative_base()


class Article(Base):
    __tablename__ = "Article"
    id = Column("id", Integer, primary_key=True, autoincrement=True)
    url = Column("url", String(300), nullable=False, unique=True)
    publisher = Column("publisher", String(50))
    datetime = Column("datetime", DateTime)
    title = Column("title", UnicodeText)
    body = Column("body", UnicodeText, nullable=False)
    category = Column("category", String(30), nullable=False)


if __name__ == "__main__":
    pass
