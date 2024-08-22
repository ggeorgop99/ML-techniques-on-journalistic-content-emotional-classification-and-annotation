import scrapy
from scrapy.http import Request
from scrapy.item import Item, Field
from scrapy.exceptions import DropItem
from scrapy.exceptions import CloseSpider
from scrapy.utils.project import get_project_settings
from rotating_proxies.middlewares import RotatingProxyMiddleware
from scrapy_user_agents.middlewares import RandomUserAgentMiddleware
import random

# URL = "https://www.skroutz.gr/c/1105/tablet.html?page=%d"
# URL = "https://www.skroutz.gr/c/1865/gaming_pontikia.html?page=%d"
URL = "https://www.skroutz.gr/c/40/kinhta-thlefwna/f/852219/Smartphones.html?page=%d"


class skroutzItem(Item):
    link = Field()


class MySpider(scrapy.Spider):
    name = "skroutz"
    allowed_domains = ["www.skroutz.gr"]

    def start_requests(self):
        print("Existing settings: %s" % self.settings.attributes.keys())
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Cookie": "AMCV_0D15148954E6C5100A4C98BC%40AdobeOrg=1176715910%7CMCIDTS%7C19271%7CMCMID%7C80534695734291136713728777212980602826%7CMCAAMLH-1665548058%7C7%7CMCAAMB-1665548058%7C6G1ynYcLPuiQxYZrsz_pkqfLG9yMXBpb2zX5dvJdYQJzPXImdj0y%7CMCOPTOUT-1664950458s%7CNONE%7CMCAID%7CNONE%7CMCSYNCSOP%7C411-19272%7CvVersion%7C5.4.0; s_ecid=MCMID%7C80534695734291136713728777212980602826; __cfruid=37ff2049fc4dcffaab8d008026b166001c67dd49-1664418998; AMCVS_0D15148954E6C5100A4C98BC%40AdobeOrg=1; s_cc=true; __cf_bm=NIDFoL5PTkinis50ohQiCs4q7U4SZJ8oTaTW4kHT0SE-1664943258-0-AVwtneMLLP997IAVfltTqK949EmY349o8RJT7pYSp/oF9lChUSNLohrDRIHsiEB5TwTZ9QL7e9nAH+2vmXzhTtE=; PHPSESSID=ddf49facfda7bcb4656eea122199ea0d",
            "If-Modified-Since": "Tue, 04 May 2021 05:09:49 GMT",
            "If-None-Match": 'W/"12c6a-5c17a16600f6c-gzip"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "TE": "trailers",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": "Mozilla/5.0 (X11; Linux i686; rv:124.0) Gecko/20100101 Firefox/124.0",
            # 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:105.0) Gecko/20100101 Firefox/105.0'
        }

        # add above headers in the request
        # initial range was 12, trying to increase it
        for i in range(1, 10):
            yield Request(
                URL % i, callback=self.parse, headers=headers
            )  # ,meta={"proxy": "http://200.29.237.154:999"})
        # self.settings = get_project_settings()
        # print("Existing settings: %s" % self.settings.attributes.keys())

    # END OF WIP
    def parse(self, response):
        urls = response.css("#sku-list")
        new_urls = urls.css(".js-sku-link")
        if not urls:
            raise CloseSpider("No more pages")
        items = skroutzItem()
        items["link"] = []

        for urls in new_urls:
            items["link"] = urls.xpath("@href").extract()
            yield items


# scrapy runspider amazon_reviews_scraping/amazon_reviews_scraping/spiders/amazon_reviews.py -o links.csv
