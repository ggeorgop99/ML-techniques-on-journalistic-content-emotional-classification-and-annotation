# -*- coding: utf-8 -*-

import pandas as pd
import scrapy
from scrapy.http import Request
from scrapy.utils.project import get_project_settings


class AmazonReviewsSpider(scrapy.Spider):
    name = "amazon_reviews"
    allowed_domains = ["skroutz.gr"]
    myBaseUrl = "https://www.skroutz.gr"
    start_urls = []
    file_name = "links.csv"

    custom_settings = {
        "BOT_NAME": "skroutz_reviews_scraping",
        "SPIDER_MODULES": ["amazon_reviews_scraping.spiders"],
        "NEWSPIDER_MODULE": "amazon_reviews_scraping.spiders",
        "ROBOTSTXT_OBEY": False,
        "CONCURRENT_REQUESTS": 1,
        "DOWNLOAD_DELAY": 2,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "CONCURRENT_REQUESTS_PER_IP": 1,
        "RETRY_ENABLED": True,
        "RETRY_TIMES": 10,
        "RETRY_HTTP_CODES": [429, 500, 502, 503, 504, 522, 524, 408],
        "COOKIES_ENABLED": False,
        "TELNETCONSOLE_ENABLED": False,
        "DEFAULT_REQUEST_HEADERS": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en",
        },
        "DOWNLOADER_MIDDLEWARES": {
            "scrapy.downloadermiddlewares.useragent.UserAgentMiddleware": None,
            "scrapy_user_agents.middlewares.RandomUserAgentMiddleware": 400,
            "scrapy.downloadermiddlewares.retry.RetryMiddleware": 90,
            # 'scrapy_proxies.RandomProxy': 100,
            # 'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 110,
        },
        # 'PROXY_LIST': 'amazon_reviews_scraping/amazon_reviews_scraping/spiders/proxies_list.txt',
        # 'PROXY_MODE': 0,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 5,
        "AUTOTHROTTLE_MAX_DELAY": 200,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 1.0,
        "AUTOTHROTTLE_DEBUG": False,
        "HTTPCACHE_ENABLED": True,
        "HTTPCACHE_EXPIRATION_SECS": 0,
        "HTTPCACHE_DIR": "httpcache",
        "HTTPCACHE_IGNORE_HTTP_CODES": [],
        "HTTPCACHE_STORAGE": "scrapy.extensions.httpcache.FilesystemCacheStorage",
    }

    def __init__(self, *args, **kwargs):
        super(AmazonReviewsSpider, self).__init__(*args, **kwargs)

        # Load and print settings
        # self.settings = get_project_settings()
        # for key, value in self.settings.items():
        #     print(f'{key}: {value}')

        for key, value in self.custom_settings.items():
            self.logger.info(f"{key}: {value}")

        self.df = pd.read_csv(self.file_name, sep="\t or ,")
        self.df.drop_duplicates(subset=None, inplace=True)
        self.df = self.df["link"].tolist()
        for i in range(len(self.df)):
            self.start_urls.append(self.myBaseUrl + self.df[i])

    def start_requests(self):
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:105.0) Gecko/20100101 Firefox/105.0",
        }

        for url in self.start_urls:
            yield Request(url, callback=self.parse, headers=headers)

    def parse(self, response):
        topic = response.css("#nav")
        top = topic.css("h2")
        title = response.css("#sku-details")
        titlee = title.css("h1")
        data = response.css("#sku_reviews_list")
        star_rating = data.css(".actual-rating")
        comments = data.css(".review-body")

        count = 0
        for review in star_rating:
            yield {
                "stars": "".join(review.xpath(".//text()").extract()),
                "comment": "".join(comments[count].xpath(".//text()").extract()),
                "topic": "".join(top.xpath(".//text()").extract()),
                "title": "".join(titlee.xpath(".//text()").extract()),
            }
            count += 1
