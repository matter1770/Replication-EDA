import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
from collections import defaultdict
import numpy as np
import time



"""
The beginning of our pipeline contains webscraping functions to show how each piece of metadata is obtained

get_catalogue_links 

Input : This function takes in a HTML text that has been processed through BeautifulSoup

Output : This function returns all the links for every section of the catalogue raisonné on a certain page as
         certain tabs reveal new section links that are not visible from the homepage
"""


def get_catalogue_links(soup):
    pages = []
    for a in soup.find_all('a', {'class': 'state-published'}):
        pages.append(a['href'])
    return pages[3:-2]


"""
get_page_links 

Input : This function takes in a URL for a section of the catalogue raisonné

Output : Returns a list of links to each artwork's RKD page which contains more information and images 
"""


def get_page_links(catalogue_url):
    html = requests.get(catalogue_url)
    soup = BeautifulSoup(html.text, 'html.parser')
    results = soup.find_all('a', {"class": "database-link"})
    links = [a['href'] for a in results]
    return links


"""
get_variable 

Input : Takes in a link to a artwork's RKD page

Output : Returns a link to an image of the artwork, the dimensions, shape, date created, title of, artist of
         and shape of the artwork
"""


def get_variable(link):
    link_html = requests.get(link)
    soup = BeautifulSoup(link_html.text, 'html.parser')
    image = get_image(soup)
    dim, shape = get_dim_and_shape(soup)
    date = get_date(soup)
    title = get_title(soup)
    artist = get_artist(soup)
    object_type = get_obj_type(soup)

    return image, dim, shape, date, title, artist, object_type


"""
get_image 

Input : Takes in a HTML text that has been processed by BeautifulSoup

Output : Returns a link to an image of the artwork
"""


def get_image(soup):
    img_link = soup.find_all('meta', {"property": "og:image"})[0]['content']
    return img_link


"""
get_dim_and_shape 

Input : Takes in a HTML text that has been processed by BeautifulSoup

Output : Returns the dimensions and shape of the artwork
"""


def get_dim_and_shape(soup):
    for d in soup.find_all('dd'):
        length = len(d.select("a[href*=VORM]"))
        if length == 1:
            text = d.text.split()
    dimensions = []
    shape = []
    shapeFlag = True
    for i in text:
        if i.isdigit():
            shapeFlag = False
        if shapeFlag:
            shape.append(i)
        else:
            dimensions.append(i)
    return ' '.join(dimensions), ' '.join(shape)


"""
get_date 

Input : Takes in a HTML text that has been processed by BeautifulSoup

Output : Returns date artwork was created
"""


def get_date(soup):
    for div in soup.find_all('div', {'class': 'fieldGroup split'}):
        row_category = div.find_all('dt')
        if 'Date' in row_category[0].text:
            dd = div.find('dd')
            text = dd.text.split()
            dates = []
            for t in text:
                if ('(' in t) or (')' in t):
                    dates.append(t)
            if len(dates) == 1:
                date = dates[0].strip('()')
                return date
            else:
                dates = [i.strip('()') for i in dates]
                date = '-'.join(dates)
                return date


"""
get_title 

Input : Takes in a HTML text that has been processed by BeautifulSoup

Output : Returns title of artwork
"""


def get_title(soup):
    text = \
    soup.find_all('div', {'class': 'fieldGroup split expandable'})[0].find_all('dd', {'class': 'expandable-content'})[
        0].text.split()
    flag = False
    title = []
    for word in text:
        if word == 'English':
            flag = True
        if word == 'Keywords':
            flag = False
        if word == 'Genre':
            flag = False
        if flag:
            title.append(word)
    return ' '.join(title[2:])


"""
get_artist 

Input : Takes in a HTML text that has been processed by BeautifulSoup

Output : Returns artist of artwork
"""


def get_artist(soup):
    for div in soup.find_all('div', {'class': 'fieldGroup split'}):
        row_category = div.find_all('dt')
        if 'Current' in row_category[0].text:
            dd = div.find('dd')
            return dd.text.strip()


"""
get_obj_type 

Input : Takes in a HTML text that has been processed by BeautifulSoup

Output : Returns type of artwork i.e. drawing, painting
"""


def get_obj_type(soup):
    for d in soup.find_all('dd'):
        length = len(d.select("a[href*=OBJALG]"))
        if length > 0:
            text = d.text.split()
            text = ' '.join(text)
    return text


"""
create_database

Input : cfg containing source of data

Output : No output but creates a DataFrame containing metadata and image links

- The default URL will be indicated in the cfg
- The pages_visited will keep track of all pages that have been visited
- The last_page flag ensures that the program does not repeat itself infinitely
- The painting_metadata DataFrame will store all links for image/metadata lookup

- First the URLs available on the page are scrapped and put into an array 'pages'
- Then it loops through each page until it reaches a page that has not been visited yet
- The page is added to the pages_visited array to indicate it has been visited and then curr_url is set to the current page
- Then all artwork links are obtained from the page and data is collected on every link and stored in a DataFrame 'painting_information'
- Then since the final page is constant in the array 'pages', if the current page is the same as the final page, the last_page flag is set to True to end the loop
- A list of images that indicate the artwork image is unavailable is kept in our config file which will then eliminate all entries with this image link
"""

def create_database(**cfg):
    curr_url = cfg['homepage']
    file = open(cfg['unavailable_images'], 'r')
    num_files = cfg['dataset_size']
    unavailable_images_url = [line.rstrip() for line in file.readlines()]
    pages_visited = []
    last_page = False

    columns = ['Image Link', 'Dimensions', 'Shape', 'Date', 'Title', 'Artist', 'Object Type']
    painting_metadata = pd.DataFrame(columns=columns)
    num_links = 0
    while not last_page:
        if num_files == -1:
            pass
        else:
            if num_links == num_files:
                break

        html = requests.get(curr_url)
        soup = BeautifulSoup(html.text, 'html.parser')
        pages = get_catalogue_links(soup)
        for p in pages:
            if p not in pages_visited:
                pages_visited.append(p)
                curr_url = p
                break
        art_links = get_page_links(curr_url)
        for link in art_links:
            if num_files == num_files:
                break
            image, dim, shape, date, title, artist, object_type = get_variable(link)
            painting_metadata = painting_metadata.append(
                {'Image Link': image, 'Dimensions': dim, 'Shape': shape, 'Date': date, 'Title': title, 'Artist': artist,
                 'Object Type': object_type}, ignore_index=True)
            time.sleep(5)
            num_links += 1
        if curr_url == pages[-1]:
            last_page = True
    for img in unavailable_images_url:
        painting_metadata = painting_metadata[painting_metadata['Image Link'] != img]

    painting_metadata = painting_metadata.drop_duplicates(subset="Image Link").dropna()
    return painting_metadata


"""
generate_list_paintings

Input : DataFrame of painting metadata
Output : Series of painting names and dates
"""


def generate_list_paintings(painting_metadata):
    return painting_metadata[painting_metadata['Object Type'] == 'painting'][['Date', 'Title']]


"""
generate_list_paintings

Input : DataFrame of painting metadata
Output : Series of painting names and dates
"""

titles = defaultdict(int)


def download_image(image_title, year, painting_metadata):
    links = painting_metadata[(painting_metadata['Title'] == image_title) and (painting_metadata['Date'] == year)][
        ['Image Link', 'Object Type', 'Title', 'Date']]

    for ix, row in links.iterrows():
        img_link = row[0]
        obj_type = row[1]
        title = row[2]
        date = row[3]

        date = date_process(date)

        if titles[title] == 0:
            titles[title] += 1
        else:
            title = title + '[{0}]'.format(titles[title])
            titles[title] += 1

        if not os.path.exists(obj_type):
            os.mkdir(obj_type + '/' + str(date))
        else:
            if not os.path.exists(obj_type + '/' + str(date)):
                os.mkdir(obj_type + '/' + str(date))

        with open(obj_type + '/' + str(date) + '/' + title + '.jpg', 'wb') as handle:
            response = requests.get(img_link, stream=True)


def download_all_image(painting_metadata):
    file_names = []
    for ix, row in painting_metadata.iterrows():
        img_link = row['Image Link']
        obj_type = row['Object Type']
        title = row['Title']
        date = row['Date']

        if type(title) == float:
            if np.isnan(title):
                title = 'No Title'

        date = date_process(date)

        if titles[title] == 0:
            titles[title] += 1
        else:
            title = title + '[{0}]'.format(titles[title])
            titles[title] += 1
        title = title.replace('"', '')
        title = title.replace('?', '')
        title = title.replace('/', '')
        title = title.replace(':', '')

        if not os.path.exists('images' + '/' + obj_type):
            os.mkdir('images' + '/' + obj_type)
            os.mkdir('images' + '/' + obj_type + '/' + str(date))
        else:
            if not os.path.exists('images' + '/' + obj_type + '/' + str(date)):
                os.mkdir('images' + '/' + obj_type + '/' + str(date))
        file_names.append('images' + '/' + obj_type + '/' + str(date) + '/' + title + '.jpeg')
        img_data = requests.get(img_link).content
        if len(title) >= 170:
            title = title[:100]
        with open('images' + '/' + obj_type + '/' + str(date) + '/' + title + '.jpeg', 'wb') as handler:
            handler.write(img_data)


def date_process(date, base=5):
    date_found = False
    if type(date) == float:
        if np.isnan(date):
            return 'Date Missing'
    if '-' not in date:
        if len(date) > 0:
            date = int(date)
            date = base * round(date / base)
    else:
        date = date.split('-')
        for i in range(len(date) - 1):
            if date[i].isdigit() and date[i + 1].isdigit() and int(date[i]) > 1500 and int(date[i + 1]) > 1500:
                date = (int(date[i]) + int(date[i + 1])) / 2
                date_found = True
                break
        if not date_found:
            for j in range(len(date)):
                if date[j].isdigit():
                    date = date[j]
                    break
        date = base * round(int(date) / base)

    return date


