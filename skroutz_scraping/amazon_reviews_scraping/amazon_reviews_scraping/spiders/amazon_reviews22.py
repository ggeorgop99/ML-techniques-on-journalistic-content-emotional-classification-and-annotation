# -*- coding: utf-8 -*-

# Importing Scrapy Library

import pandas as pd
import scrapy
from scrapy.utils.project import get_project_settings
from scrapy.http import Request


# Creating a new class to implement Spider
class AmazonReviewsSpider(scrapy.Spider):
    file_name = "links.csv"

    # Spider name
    name = "skroutz22"

    # Domain names to scrape
    allowed_domains = ["skroutz.gr"]

    # Base URL
    myBaseUrl = "https://www.skroutz.gr"
    start_urls = []

    df = pd.read_csv(file_name, sep="\t or ,")
    df.drop_duplicates(subset=None, inplace=True)
    print(df.head())
    df = df["link"].tolist()
    # Creating list of urls to be scraped by appending page number at the end of base url
    for i in range(1, len(df)):
        start_urls.append(myBaseUrl + df[i])

    def start_requests(self):
        self.settings = get_project_settings()
        print(self.settings.attributes.keys())
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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:105.0) Gecko/20100101 Firefox/105.0",
        }

        # add above headers in the request
        for i in range(len(self.df)):
            yield Request(self.start_urls[i], callback=self.parse, headers=headers)

    # Defining a Scrapy parser
    def parse(self, response):
        topic = response.css("#nav")

        # collecting topic
        top = topic.css("h2")

        title = response.css("#sku-details")
        titlee = title.css("h1")

        data = response.css("#sku_reviews_list")

        # Collecting product star ratings
        star_rating = data.css(".actual-rating")

        # Collecting user reviews
        comments = data.css(".review-body")

        # vote = data.css('.review-rate')
        count = 0

        # Combining the results
        for review in star_rating:
            yield {
                "stars": "".join(review.xpath(".//text()").extract()),
                "comment": "".join(comments[count].xpath(".//text()").extract()),
                #'vote': ''.join(vote[count].xpath("//div[@class='review-rate']/text()").extract()),
                "topic": "".join(top.xpath(".//text()").extract()),
                "title": "".join(titlee.xpath(".//text()").extract()),
            }
            count = count + 1


# scrapy runspider amazon_reviews_scraping/amazon_reviews_scraping/spiders/amazon_reviews22.py -o dirtyreviews.csv
