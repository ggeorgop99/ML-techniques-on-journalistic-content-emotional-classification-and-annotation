# ML-techniques-on-journalistic-content-emotional-classification-and-annotation

Repo for the code associated to my diploma thesis for the department of Electrical and Computer Engineering, of the faculty of engineering, of the Aristotle University of Thessaloniki.

The code provided attempts to do two tasks:

1. create a big sentiment-wise anotated dataset in greek, scraped from the online shopping website "skroutz"

2. use various algorithmic techniques, ML and non-ML, to analyze the sentiment of written text in greek.

mysenti.py is the code of the python implementation of <a href="http://sentistrength.wlv.ac.uk/" target="_blank">sentistrength algorithm</a>., reviews.csv are the reviews (2800), stars.csv are the stars of every review, in folder finallexformysenti there are the lexicons and in folder dataset there are a .csv with results from crawlers (dirtyreviews), a .csv with those results cleared (reviewstars) and a .csv with results.

Inside the folder skroutz_scraping and subfolder spiders there are the two crawlers, amazon_reviews.py and amazon_reviews22.py, and various scripts for the dataset creation process.

They run with the following commands:
</br>
scrapy runspider amazon_reviews_scraping/amazon_reviews_scraping/spiders/amazon_reviews.py -o links.csv</br>
and</br>
scrapy runspider amazon_reviews_scraping/amazon_reviews_scraping/spiders/amazon_reviews22.py -o dirtyreviews.csv</br>
FROM THE FOLDER amazon_reviews_scraping

Inside the neuralnet folder are the all necessary scripts and files for training the machine learning models and running them, while also one can find there all the related results, graphs and already trained models.

There is function clearfiles() inside mysenti.py that makes reviewstars.csv that is suitable for mysentistrength and function splitfiles() that makes reviews.csv and stars.csv. Note that if there is not a dirtyreviews.csv inside the folder, clearfiles() does not run.
