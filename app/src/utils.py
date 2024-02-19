from statistics import pstdev
from itertools import product
from typing import Tuple

import streamlit as st
import pandas as pd
import requests

from . import constants
from .types import APIResponse, InstagramResponse, InstagramVenue, HttpStatus


# Functions
def query_instagram(lat: float, lng: float, cookies: str) -> APIResponse | None:
    """Queries Instagram location API

    Args:
        lat (float): area latitude
        lng (float): area longitude
        cookies (str): personal Instagram cookies

    Returns:
        InstagramResponse | None:
    """
    params = {"latitude": lat, "longitude": lng}  # __a supports pagination
    headers = {"Cookie": cookies}
    try:
        response = requests.get(
            constants.INSTAGRAM_URL,
            params=params,
            headers=headers,
            timeout=constants.INSTAGRAM_TIMEOUT,
        )
        print(response.status_code)
        if response.status_code == HttpStatus.ok_200.value:
            try:
                body = response.json()
                body["venues"] = ""
                return APIResponse(HttpStatus.ok_200, InstagramResponse(**body))
            # if cookies are invalid the response code is still 200
            except (ValueError, TypeError) as e:
                print(f"No values returned for params: {params}: {e}")
                return APIResponse(HttpStatus.bad_request_400, {})
        if response.status_code == HttpStatus.too_many_requests_429.value:
            print("Too many requests for 1 hour. 200 per hour limit")
            return APIResponse(HttpStatus.too_many_requests_429, response.json())
    except requests.exceptions.ConnectionError as e:
        print(f"Connection failed for params: {params}: {e}")
    except requests.exceptions.Timeout:
        print(f"Connections timed out after {constants.INSTAGRAM_TIMEOUT} seconds")


