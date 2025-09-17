from bs4 import BeautifulSoup
import requests
import json


r = requests.get('https://www.profesia.sk/praca/bratislava/?gclid=CjwKCAjwlaTGBhANEiwAoRgXBQCat9uF5ds22jyHqyuEA7GMdT-EKZxCSSSStzAjm2WdrpL8eBzaThoC2r4QAvD_BwE&radius=radius30&search_anywhere=python&sort_by=relevance&utm_campaign=pfx-profesia-sk-gsn-reaction-brigady&utm_medium=cpc&utm_source=google')

soup = BeautifulSoup(r.text, 'html.parser')
print(soup.find("title").text)
print(soup.find("h1").text)

alltags = soup.find_all("span", class_ = "title")
for tags in alltags:
   if "Python" in tags.text:
       print(tags.text.strip())

vacancy = []
for tags in alltags:
    vacancy.append(tags.text.strip())

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(vacancy, f, ensure_ascii=False, indent=4)

alltagsinside = soup.find_all("a")
for links in alltagsinside:
    if links.get("href") != None:
        if "offer" in links.get("href"):
                print(f"{links.text.strip()}")