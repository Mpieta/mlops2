import os
import requests
from typing import Annotated
from fastmcp import FastMCP

api_key = os.getenv("OPENWEATHER_API_KEY")
mcp = FastMCP("OpenWeatherMap Service")


def get_coords(city: str):
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={api_key}"
    resp = requests.get(url).json()
    if not resp:
        raise ValueError(f"City '{city}' not found.")
    return resp[0]['lat'], resp[0]['lon']


@mcp.tool(description="Get daily weather forecast for up to 16 days.")
def get_daily_forecast(
        city: Annotated[str, "City name"],
        days: Annotated[int, "Number of days (max 16)"] = 5
) -> str:
    try:
        lat, lon = get_coords(city)
        url = f"https://api.openweathermap.org/data/2.5/forecast/daily?lat={lat}&lon={lon}&cnt={days}&units=metric&appid={api_key}"
        resp = requests.get(url)
        data = resp.json()

        if str(data.get("cod")) != "200":
            return f"Error fetching forecast: {data.get('message')}"

        report = [f"Forecast for {city}:"]
        for day in data['list']:
            date = day.get('dt')
            temp = day['temp']
            desc = day['weather'][0]['description']
            report.append(f"Date TS {date}: {desc}, Temp: {temp['day']} C (Min: {temp['min']}, Max: {temp['max']})")

        return "\n".join(report)
    except Exception as e:
        return f"Weather tool error: {str(e)}"


@mcp.tool(description="Get monthly average weather statistics (climate).")
def get_monthly_average(
        city: Annotated[str, "City name"],
        month: Annotated[int, "Month number (1-12)"]
) -> str:
    try:
        lat, lon = get_coords(city)
        url = f"https://history.openweathermap.org/data/2.5/aggregated/month?lat={lat}&lon={lon}&month={month}&appid={api_key}"
        resp = requests.get(url)
        data = resp.json()

        if str(data.get("cod")) != "200" and "result" not in data:
            return f"Error fetching stats: {data.get('message')}"

        res = data['result']
        temp = res['temp']
        return (f"Climate averages for {city} in month {month}:\n"
                f"Mean Temp: {temp['mean'] - 273.15:.1f} C\n"
                f"Record High: {temp['record_max'] - 273.15:.1f} C\n"
                f"Record Low: {temp['record_min'] - 273.15:.1f} C\n"
                f"Precipitation: {res['precipitation']['mean']:.1f}mm")
    except Exception as e:
        return f"Climate tool error: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8004)