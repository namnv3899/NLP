# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
from itemloaders.processors import Join, MapCompose, TakeFirst
from w3lib.html import remove_tags


def clean_text(text):
    return remove_tags(text).strip()


class ArticleItem(scrapy.Item):
    url = scrapy.Field(output_processor=TakeFirst())
    publisher = scrapy.Field(output_processor=TakeFirst())
    datetime = scrapy.Field(output_processor=TakeFirst())
    title = scrapy.Field(output_processor=TakeFirst())
    body = scrapy.Field(
        input_processor=MapCompose(clean_text), output_processor=Join("\n")
    )
    category = scrapy.Field(
        input_processor=MapCompose(str.lower), output_processor=TakeFirst()
    )

    def __repr__(self):
        item = self.__dict__["_values"]
        return repr(
            {
                "title": item["title"],
                "category": item["category"],
                "datetime": item["datetime"],
            }
        )


if __name__ == "__main__":
    pass
