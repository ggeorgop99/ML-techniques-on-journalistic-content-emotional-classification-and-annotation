# -*- coding: utf-8 -*-

# Scrapy settings for amazon_reviews_scraping project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

# [WIP] add the below to maybe fix 403 error

# # Allow 404 errors
# HTTPERROR_ALLOWED_CODES = [404, 429]

# # Enable retries for 429 errors
# RETRY_HTTP_CODES = [429]
# RETRY_TIMES = 10  # Number of times to retry
USER_AGENT = "quotesbot (+http://www.yourdomain.com)"
USER_AGENT = "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36"
# end of [WIP]

BOT_NAME = "skroutz_reviews_scraping"

SPIDER_MODULES = ["amazon_reviews_scraping.spiders"]
NEWSPIDER_MODULE = "amazon_reviews_scraping.spiders"


# Crawl responsibly by identifying yourself (and your website) on the user-agent
USER_AGENT = "AUTH-SKG MASTER THESIS BOT"  # (+http://www.google.com/adsbot.html)'#amazon_reviews_scraping (+http://www.yourdomain.com)'

# Obey robots.txt rules
ROBOTSTXT_OBEY = False

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 1

# Configure a delay for requests for the same website (default: 0)
# See https://docs.scrapy.org/en/latest/topics/settings.html#download-delay
# See also autothrottle settings and docs
DOWNLOAD_DELAY = 1000
# The download delay setting will honor only one of:
CONCURRENT_REQUESTS_PER_DOMAIN = 1
CONCURRENT_REQUESTS_PER_IP = 0

# AUTOTHROTTLE_ENABLED = True
# AUTOTHROTTLE_START_DELAY = 5  # Initial download delay
# AUTOTHROTTLE_MAX_DELAY = 60  # Maximum download delay in case of high latencies
# AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0  # Average number of requests to be sent in parallel

# # Enable randomization of download delay to avoid detection
# RANDOMIZE_DOWNLOAD_DELAY = True

# # Configure user agent middleware to rotate user agents
# DOWNLOADER_MIDDLEWARES = {
#     'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
#     'scrapy_user_agents.middlewares.RandomUserAgentMiddleware': 400,
#     'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
#     'scrapy_proxies.RandomProxy': 100,
#     'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 110,
# }


# # Use a larger pool of proxies
# ROTATING_PROXY_LIST = [
#     'proxy1:port',
#     'proxy2:port',
#     # Add more proxies as needed
# ]

# Disable cookies (enabled by default)
# COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
# TELNETCONSOLE_ENABLED = False

# Override the default request headers:
# DEFAULT_REQUEST_HEADERS = {
#   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
#   'Accept-Language': 'en',
# }

# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
# SPIDER_MIDDLEWARES = {
#     'amazon_reviews_scraping.middlewares.AmazonReviewsScrapingSpiderMiddleware': 543,
# }

# # Enable or disable downloader middlewares
# # See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
# DOWNLOADER_MIDDLEWARES = {
#     'amazon_reviews_scraping.middlewares.AmazonReviewsScrapingDownloaderMiddleware': 350,
# }

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
# EXTENSIONS = {
#    'scrapy.extensions.telnet.TelnetConsole': None,
# }

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
# ITEM_PIPELINES = {
#'amazon_reviews_scraping.pipelines.AmazonReviewsScrapingPipeline': 300,
#    'amazon_reviews_scraping.pipelines.DuplicatesPipeline': 300,

# }

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
# AUTOTHROTTLE_ENABLED = True
# The initial download delay
# AUTOTHROTTLE_START_DELAY = 5
# The maximum download delay to be set in case of high latencies
# AUTOTHROTTLE_MAX_DELAY = 60
# The average number of requests Scrapy should be sending in parallel to
# each remote server
# AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# Enable showing throttling stats for every response received:
# AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
# HTTPCACHE_ENABLED = True
# HTTPCACHE_EXPIRATION_SECS = 0
# HTTPCACHE_DIR = 'httpcache'
# HTTPCACHE_IGNORE_HTTP_CODES = []
# HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'
