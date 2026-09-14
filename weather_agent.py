"""LangChain single-agent weather search.

Usage:
    pip install langchain langchain-openai langgraph python-dotenv requests
    # Create a .env file with:
    #   OPENAI_API_KEY=sk-your-openai-key
    #   OPENWEATHERMAP_API_KEY=your-openweathermap-key
    python weather_agent.py
"""

import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
API_KEY = os.environ["OPENWEATHERMAP_API_KEY"]


@tool
def get_current_weather(city: str) -> str:
    """Get the current weather for a city. Input should be a city name,
    optionally with country code, e.g. 'Tokyo' or 'Paris,FR'."""
    try:
        resp = requests.get(
            BASE_URL,
            params={"q": city, "appid": API_KEY, "units": "metric"},
            timeout=10,
        )
        data = resp.json()

        if resp.status_code != 200:
            return f"Error: {data.get('message', 'unknown error')}"

        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        wind = data["wind"]["speed"]
        country = data["sys"]["country"]

        return (
            f"Current weather in {data['name']}, {country}: "
            f"{weather}, {temp}°C (feels like {feels_like}°C), "
            f"humidity {humidity}%, wind {wind} m/s."
        )
    except requests.RequestException as e:
        return f"Failed to reach weather service: {e}"


@tool
def get_weather_forecast(city: str) -> str:
    """Get a short weather forecast summary for a city (next ~24h)."""
    try:
        resp = requests.get(
            FORECAST_URL,
            params={"q": city, "appid": API_KEY, "units": "metric", "cnt": 8},
            timeout=10,
        )
        data = resp.json()

        if resp.status_code != 200:
            return f"Error: {data.get('message', 'unknown error')}"

        entries = []
        for item in data["list"]:
            entries.append(
                f"{item['dt_txt']}: {item['weather'][0]['description']}, "
                f"{item['main']['temp']}°C"
            )
        return "Forecast:\n" + "\n".join(entries)
    except requests.RequestException as e:
        return f"Failed to reach weather service: {e}"


# --- Agent setup ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)

agent = create_agent(
    llm,
    tools=[get_current_weather, get_weather_forecast],
    system_prompt=(
        "You are a weather assistant. Use the available tools to look up "
        "current conditions or forecasts. Answer concisely in natural language."
    ),
)

if __name__ == "__main__":
    question = input("Ask about the weather: ")

    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    print("\n--- Final answer ---")
    print(result["messages"][-1].content)
