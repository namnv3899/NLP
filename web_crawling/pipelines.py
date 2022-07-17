# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


from datetime import datetime

# useful for handling different item types with a single interface
from scrapy.exceptions import DropItem

from models import ArticleDB
from utils import get_logger

logobj = get_logger(__name__)


class PreProcessPipeline:
    def _convert_datetime(self, text):
        if text is None:
            return None
        text = text.replace("GMT+7", "").strip()
        return datetime.strptime(text, "%d/%m/%Y %H:%M")

    def process_item(self, item, spider):
        item["datetime"] = self._convert_datetime(item.get("datetime"))
        return item


class DBInsertingPipeline:
    def __init__(self) -> None:
        self.db = ArticleDB()

    def process_item(self, item, spider):
        session = self.db.create_session()
        new_article = self.db.insert_article(
            session,
            {
                "url": item["url"],
                "publisher": item["publisher"],
                "datetime": item["datetime"],
                "title": item["title"],
                "body": item["body"],
                "category": item["category"],
            },
        )
        if new_article is None:
            session.close()

            logobj.info(
                f'[CRAWLING] Drop item from url: {item["url"]}, datetime: {item["datetime"]}'
            )

            raise DropItem()

        logobj.info(
            f'[CRAWLING] Import item from url: {item["url"]}, datetime: {item["datetime"]} to database'
        )

        session.close()
        return item
