

from typing import List
from pydantic import BaseModel
from robyn.types import Body
from models.requests.tweet_request import TweetUserData


class RequestDigitalDNA(BaseModel, Body):
    socmed_data: TweetUserData
    unique_id: List[str]
    title: List[str]

class RequestDigitalDNAImage(BaseModel, Body):
    title: str