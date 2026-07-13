import requests


def login_as_guest(base_url: str, username: str) -> tuple[requests.Session, dict]:
    session = requests.Session()
    url = f"{base_url.rstrip('/')}/api/auth/guest"
    response = session.post(url, json={"username": username})
    if response.status_code != 201:
        raise ValueError(f"Guest login failed with status {response.status_code}: {response.text}")
    user_info = response.json()
    return session, user_info
