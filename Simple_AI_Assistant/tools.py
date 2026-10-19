from datetime import datetime
import requests

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S");

# print(get_current_time())

def get_city_weather(city):
    url=f"https://wttr.in/{city}?format=3"
    response=requests.get(url,timeout=10)
    return response.text


def get_user_city():
    return "Noida"

TOOLS=[
    {
        "type":"function",
        "function":{
            "name":"get_current_time",
            "description":"Returns the current date and time as a formatted string.",
            "parameters":{
                "type":"object",
                "properties":{},
            }
        }
    },

    {
        "type":"function",
        "function":{
            "name":"get_city_weather",
            "description":"Returns the current weather for a given city.",
            "parameters":{
                "type":"object",
                "properties":{
                    "city":{
                        "type":"string",
                        "description":"City to get weather for."
                    }
                },    
            }
        }
    },

    {
        "type":"function",
        "function":{
            "name":"get_user_city",
            "description":"Returns the current city where the user lives.",
            "parameters":{
                "type":"object",
                "properties":{},
            }
        }
    }

]

FUNCTION_NAME_TO_FUNCTION_MAP={
    "get_current_time":get_current_time,
    "get_city_weather":get_city_weather,
    "get_user_city":get_user_city,
}
