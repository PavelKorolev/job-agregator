from bs4 import BeautifulSoup
import requests
import json
import re
import os

from fastapi import FastAPI

app = FastAPI()

MODEL_URL = os.getenv(
    "MODEL_URL",
    "http://host.docker.internal:11434/v1/chat/completions"
)

MODEL_NAME = "deepseek-r1:1.5b"


def get_soup(url: str) -> BeautifulSoup:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


@app.get("/")
def read_root():
    start = "https://www.profesia.sk"
    search_url = (
        "https://www.profesia.sk/praca/bratislava/"
        "?search_anywhere=python"
    )

    soup = get_soup(search_url)

    alltags = soup.find_all(
        "span",
        class_="title",
        string=lambda x: "Python" in x if x else False
    )

    vacancies = []

    for tags in alltags[:3]:  # ограничение специально
        title = tags.text.strip()

        location_tag = tags.find_next("span", class_="job-location")
        location = location_tag["title"].strip() if location_tag else "не указано"

        salary_tag = tags.find_next("span", class_="label")
        salary = salary_tag.text.strip() if salary_tag else "не указано"

        a_tag = tags.find_next("a", href=True)
        vacancy_url = start + a_tag["href"]

        vacancy_page = get_soup(vacancy_url)
        desc = vacancy_page.find("div", class_="job-info details-section")
        description = desc.get_text(" ", strip=True) if desc else "не указано"

        vacancies.append({
            "title": title,
            "location": location,
            "salary": salary,
            "description": description
        })

    vacancies_json = json.dumps(vacancies, ensure_ascii=False)

    messages = [
        {
            "role": "user",
            "content": f"""
You are given a list of job vacancies in JSON format.

For EACH vacancy extract ONLY technical and professional skills
that are EXPLICITLY mentioned in the description.

Do NOT invent skills.
Do NOT infer skills.
If no skills are mentioned, return an empty list.

STRICT RULES:
- Output ONLY valid JSON
- No explanations
- No markdown
- No extra text

Input vacancies:
{vacancies_json}

Output format:
{{
  "vacancies": [
    {{
      "title": "<same title as input>",
      "skills": ["skill1", "skill2"]
    }}
  ]
}}
"""
        }
    ]

    response = requests.post(
        MODEL_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "model": MODEL_NAME,
            "messages": messages
        }),
        timeout=300
    )

    data = response.json()
    raw_content = data["choices"][0]["message"]["content"]

    clean_content = re.sub(
        r"<think>.*?</think>",
        "",
        raw_content,
        flags=re.DOTALL
    )

    result = json.loads(clean_content)

    return result