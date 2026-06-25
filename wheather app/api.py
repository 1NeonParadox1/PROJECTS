import requests


def get_coordinates(city):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"

    response = requests.get(url).json()

    if "results" not in response:
        return None

    latitude = response["results"][0]["latitude"]
    longitude = response["results"][0]["longitude"]

    return latitude, longitude


def get_current_weather(city):

    coords = get_coordinates(city)

    if coords is None:
        return None

    latitude, longitude = coords

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}"
        f"&longitude={longitude}"
        f"&current="
        f"temperature_2m,"
        f"relative_humidity_2m,"
        f"wind_speed_10m,"
        f"surface_pressure,"
        f"visibility,"
        f"uv_index"
    )

    response = requests.get(url)

    return response.json()