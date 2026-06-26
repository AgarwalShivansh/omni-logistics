import math
import requests


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * \
        math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def fetch_live_weather_alert(lat: float, lon: float) -> str:
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        temp = data['current']['temperature_2m']
        wind = data['current']['wind_speed_10m']

        if wind > 30:
            return f"SEVERE WEATHER ALERT: High winds at {wind} km/h detected. Expect heavy disruptions."
        elif wind > 15:
            return f"Moderate weather alert. Winds at {wind} km/h, temperature {temp}C."
        else:
            return f"Clear weather at destination. Winds calm at {wind} km/h, temp {temp}C."
    except Exception as e:
        return "Failed to fetch live weather. Assuming neutral conditions."
