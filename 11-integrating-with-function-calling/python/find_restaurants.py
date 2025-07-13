import requests
import os
import dotenv
import json
from openai import AzureOpenAI

GOOGLE_API_KEY = ""
GOOGLE_API_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"

def find_restaurants(city: str, cuisine: str):
    query = f"{cuisine} restaurants in {city}"
    params = {
        "query": query,
        "key": GOOGLE_API_KEY
    }

    response = requests.get(GOOGLE_API_URL, params=params)
    data = response.json()

    results = []
    for place in data.get("results", [])[:5]:
        name = place["name"]
        address = place.get("formatted_address", "No address")
        rating = place.get("rating", "N/A")
        place_id = place["place_id"]

        # Call Places Details API to get opening hours
        details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        details_params = {
            "place_id": place_id,
            "fields": "opening_hours",
            "key": GOOGLE_API_KEY
        }
        details_response = requests.get(details_url, params=details_params)
        details_data = details_response.json()

        result_data = details_data.get("result", {})
        opening_info = result_data.get("opening_hours", {})

        # Get whether it's open now
        open_now = opening_info.get("open_now")
        open_status = "Open now ✅" if open_now else "Closed ❌" if open_now is not None else "Unknown status"

        # Get weekday hours (like Mon–Sun schedule)
        weekday_hours = opening_info.get("weekday_text", [])
        hours_text = "\n".join(weekday_hours) if weekday_hours else "No hours listed"

        result = (
            f"{name} - Rating: {rating} - Address: {address}\n"
            f"Status: {open_status}\n"
            f"Hours:\n{hours_text}"
        )
        results.append(result)

    return results if results else ["No restaurants found."]



if __name__ == "__main__":
    dotenv.load_dotenv()
    client = AzureOpenAI(
        api_key=os.environ['AZURE_OPENAI_API_KEY'],
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        api_version="2023-10-01-preview"
    )

    deployment = os.environ['AZURE_OPENAI_DEPLOYMENT']

    messages = [{"role": "user", "content": "Find me some Italian restaurants in Bangalore that are open now ."}]

    functions = [
        {
            "name": "find_restaurants",
            "description": "Finds restaurants of a specific cuisine in a given Indian city using Google Places API",
            "parameters": {
                "type": "object",
                "properties": {
                "city": {
                    "type": "string",
                    "description": "The city where the user wants to find restaurants"
                },
                "cuisine": {
                    "type": "string",
                    "description": "The type of cuisine to search for"
                }
                },
                "required": ["city", "cuisine"]
            }
        }
    ]

    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        functions=functions,
        function_call="auto"  # Automatically call the function if needed
    )

    response_message = response.choices[0].message

    if response_message.function_call.name:
        print("Recommended Function call:")
        print(response_message.function_call.name)
        print()

        # Call the function. 
        function_name = response_message.function_call.name

        available_functions = {
            "find_restaurants": find_restaurants,
        }
        function_to_call = available_functions[function_name] 

        function_args = json.loads(response_message.function_call.arguments)
        function_response = function_to_call(**function_args)

        print("Output of function call:")
        print(function_response)
        print(type(function_response))


        # Add the assistant response and function response to the messages
        messages.append( # adding assistant response to messages
            {
                "role": response_message.role,
                "function_call": {
                    "name": function_name,
                    "arguments": response_message.function_call.arguments,
                },
                "content": None
            }
        )
        messages.append( # adding function response to messages
            {
                "role": "function",
                "name": function_name,
                "content": json.dumps(function_response),  # Convert list to string
            }
        )

    print("Messages in next request:")
    print(messages)
    print()

    second_response = client.chat.completions.create(
        messages=messages,
        model=deployment,
        function_call="auto",
        functions=functions,
        temperature=0
    )


    print(second_response.choices[0].message.content)

