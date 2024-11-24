# import requests


# def list_customers():
#     url = "https://api.thinkimmo.com/organization/integrations/customer"

#     headers = {
#         'Content-Type': 'application/json',
#         'AuthorizationApi': '133bd0b5-cd49-4029-8b0e-e5154597a086'
#     }

#     response = requests.get(url, headers=headers)
#     return response.json()

# print(list_customers())



# ####

# import requests
# import json

# def create_search_alert(customer_id, alert_name, alert_settings, filter_criteria):
#     """
#     Create a search alert for a customer with specific filter criteria.
    
#     Args:
#         customer_id (str): The ID of the customer.
#         alert_name (str): The name of the alert.
#         alert_settings (dict): Settings for the alert (e.g., sending_type).
#         filter_criteria (dict): The filter parameters for housing data.
    
#     Returns:
#         dict: The API response containing alert details.
#     """
#     url = "https://api.thinkimmo.com/organization/integrations/alert"

#     payload = json.dumps({
#         "customer_id": customer_id,
#         "alertSettings": alert_settings,
#         "filter": filter_criteria
#     })

#     headers = {
#         'Content-Type': 'application/json',
#         'AuthorizationApi': '133bd0b5-cd49-4029-8b0e-e5154597a086'
#     }

#     response = requests.post(url, headers=headers, data=payload)
#     return response.json()

# if __name__ == "__main__":
#     alert_settings = {
#         "name": "Available Homes in Berlin",
#         "sending_type": 3,  # 3 times daily
#         "send_price_reduced": True,
#         "send_related_objects": False
#     }

#     filter_criteria = {
#         "sortBy": "publishDate,desc",
#         "from": 0,
#         "size": 50,  # Number of results per alert
#         "active": True,
#         "grossReturnAnd": False,
#         "allowUnknown": False,
#         "ownCapital": "15",
#         "ownCapitalInPercent": "true",
#         "ownCapitalAdditionalCost": False,
#         "managementCostInPercent": "true",
#         "renovationCostInPercent": "true",
#         "renovationCost": 0,
#         "interestRate": "2.5",
#         "repaymentRate": 2,
#         "managementCost": 30,
#         "additionalPurchaseCost": 1,
#         "foreClosure": False,
#         "leasehold": False,
#         "buyingPriceFrom": 250000,
#         "buyingPriceTo": 750000,
#         "geoSearches": [
#             {
#                 "geoSearchType": "radius",
#                 "lat": 52.5200,      # Latitude for Berlin
#                 "lng": 13.4050,      # Longitude for Berlin
#                 "radius": "10km"
#             }
#         ],
#         "type": "HOUSEBUY",
#         "buildingType": "MULTI_FAMILY_HOUSE,SINGLE_FAMILY_HOUSE",
#         "sqmFrom": 80,
#         "sqmTo": 250,
#         "roomsFrom": 3,
#         "roomsTo": 10,
#         "pricePerSqmFrom": 3000,
#         "pricePerSqmTo": 8000,
#         "interiorQuality": "NORMAL,LUXURY",
#         "floorFrom": 1,
#         "floorTo": 5,
#         "balcony": True,
#         "numberOfParkingSpacesFrom": 1,
#         "numberOfParkingSpacesTo": 3
#     }

#     customer_id = "0"

#     response = create_search_alert(customer_id, "Available Homes in Berlin", alert_settings, filter_criteria)
#     print(json.dumps(response, indent=4))


import requests
import json

url = "https://api.thinkimmo.com/organization/customer/onboarding_widget"

payload = json.dumps({
  "filter": {
    "sortBy": "publishDate,desc",
    "from": None,
    "size": 20,
    "active": True,
    "grossReturnAnd": False,
    "allowUnknown": False,
    "ownCapital": "10",
    "ownCapitalInPercent": "true",
    "ownCapitalAdditionalCost": False,
    "managementCostInPercent": "true",
    "renovationCostInPercent": "true",
    "renovationCost": None,
    "interestRate": "3",
    "repaymentRate": 2,
    "managementCost": 35,
    "additionalPurchaseCost": 2,
    "foreClosure": False,
    "leasehold": False,
    "buyingPriceFrom": 300000,
    "geoSearches": [
      {
        "geoSearchQuery": "Berlin",
        "geoSearchType": "city",
        "region": "Berlin",
        "lat": 52.5170365,
        "lng": 13.3888599
      }
    ],
    "type": "HOUSEBUY",
    "buildingType": "MULTI_FAMILY_HOUSE,SINGLE_FAMILY_HOUSE"
  },
  "alertSettings": {
    "name": "Einfamilienhäuser in Berlin",
    "sending_type": 3,
    "send_price_reduced": True,
    "send_related_objects": False
  },
  "accountSettings": {
    "firstName": "Fabian",
    "lastName": "Lurz",
    "email": "fabian+12312ads12312a13@thinkimmo.com",
    "phoneNumber": "01638687140",
    "type": "OWN_HOME",
    "salutation": "THY"
  }
})
headers = {
  'AuthorizationWidget': '133bd0b5-cd49-4029-8b0e-e5154597a086',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
