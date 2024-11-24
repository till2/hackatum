import requests
from typing import Optional, Dict, Any

class ImmobilienScout24APIClient:
    """
    Client to interact with the ImmobilienScout24 API.
    """

    BASE_URL = "https://api.immobilienscout24.de"

    def __init__(self, username: str, password: str, client_id: str, client_secret: str):
        """
        Initialize the API client with authentication credentials.
        """
        self.username = username
        self.password = password
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = self.authenticate()

    def authenticate(self) -> str:
        """
        Authenticate with the API and retrieve an access token.
        """
        auth_url = f"{self.BASE_URL}/oauth/token"
        payload = {
            'grant_type': 'password',
            'username': self.username,
            'password': self.password,
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        response = requests.post(auth_url, data=payload)
        if response.status_code == 200:
            token_info = response.json()
            return token_info['access_token']
        else:
            raise Exception(f"Authentication failed: {response.text}")

    def get_realestate(self, username: str, realestate_id: str, external_id: Optional[str] = None, use_new_energy_source: bool = False, response_format: str = 'JSON') -> Any:
        """
        Retrieve a real estate object by its ID or external ID.
        
        :param username: The username for authentication.
        :param realestate_id: The unique scout object ID.
        :param external_id: (Optional) The customer-defined external ID.
        :param use_new_energy_source: (Optional) Whether to use the new energy source parameters.
        :param response_format: The desired response format ('JSON' or 'XML').
        :return: The real estate data in the specified format.
        """
        if external_id:
            id_param = f"ext-{external_id}"
        else:
            id_param = realestate_id

        url = f"{self.BASE_URL}/offer/v1.0/user/{username}/realestate/{id_param}"
        headers = {
            'Authorization': f"Bearer {self.access_token}",
            'Accept': 'application/json' if response_format.upper() == 'JSON' else 'application/xml'
        }
        params = {}
        if use_new_energy_source:
            params['usenewenergysourceenev2014values'] = 'true'

        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            if response_format.upper() == 'JSON':
                return response.json()
            else:
                return response.text  # XML response as string
        else:
            raise Exception(f"Failed to retrieve real estate data: {response.text}")

def main():
    client = ImmobilienScout24APIClient(
        username="-",
        password="-",
        client_id="-",
        client_secret="-"
    )

    try:
        # Retrieve real estate by internal ID in JSON format
        realestate_json = client.get_realestate(
            username="your_username",
            realestate_id="74719707",
            use_new_energy_source=True,
            response_format="JSON"
        )
        print("Real Estate Data (JSON):", realestate_json)

        # Retrieve real estate by external ID in XML format
        realestate_xml = client.get_realestate(
            username="your_username",
            realestate_id="",
            external_id="441",
            use_new_energy_source=True,
            response_format="XML"
        )
        print("Real Estate Data (XML):", realestate_xml)

    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    main()
