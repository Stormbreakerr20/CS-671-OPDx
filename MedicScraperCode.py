from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

from bs4 import BeautifulSoup
import time

service = Service(executable_path="chromedriver.exe")
driver = webdriver.Chrome(service=service)


# The below method is to by pass Captcha Protection on the website
driver.execute_script(f"window.open('https://www.nejm.org/case-challenges','_blank');") # open page in new tab
time.sleep(5) # wait until page has loaded
driver.switch_to.window(window_name=driver.window_handles[0])   # switch to first tab
driver.close() # close first tab
driver.switch_to.window(window_name=driver.window_handles[0] )  # switch back to new tab
time.sleep(2)
driver.get("https://google.com")
time.sleep(2)
driver.get('https://www.nejm.org/case-challenges') # this should pass cloudflare captchas now
 
# Get all Case Challenges and their links
case_challenges = driver.find_elements(By.CLASS_NAME, "issue-item_title-link")
links = []
for case_challenges_link in case_challenges:
    links.append(case_challenges_link.get_attribute("href"))

time.sleep(10)

# Create a dictionary to store the case studies
d = {}

for link in links:

    driver.execute_script(f"window.open('{link}','_blank');") # open page in new tab
    time.sleep(5) # wait until page has loaded
    driver.switch_to.window(window_name=driver.window_handles[0])   # switch to first tab
    driver.close() # close first tab
    driver.switch_to.window(window_name=driver.window_handles[0] )  # switch back to new tab
    time.sleep(2)
    driver.get("https://google.com")
    time.sleep(2)
    driver.get(link) # this should pass cloudflare captchas now

    print(link)
    time.sleep(10)
    html_content = driver.page_source

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")

    # Find the section with id "sec-2"
    section_sec_2 = soup.find("section", id="sec-2")

    # Extract all paragraphs within the section
    paragraphs = section_sec_2.find_all("div", {"role": "paragraph"})

    content = ""
    # Extract text from each paragraph
    for paragraph in paragraphs:
        content = content + " " + paragraph.text.strip()

    # title
    title = soup.find("h1", property="name").text.strip()

    d[title] = content
    time.sleep(10)

# Write the dictionary to a file
def write_dict_to_file(dictionary, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        i = 1
        for title, case in dictionary.items():
            file.write(f"Case Study {i}\n")
            file.write(f"Title: {title}\n")
            file.write(f"Case Study: {case}\n\n")
            file.write("<------------------------------------------------------------------------------------------->\n")
            file.write("<------------------------------------------------------------------------------------------->\n\n\n\n")
            i += 1


write_dict_to_file(d, "Case_Studies.txt")


