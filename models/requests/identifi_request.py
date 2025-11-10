
from typing import Optional
from pydantic import BaseModel
from robyn.types import Body

from models.requests.tweet_request import TweetUserData

class RequestIdentifiScore(TweetUserData, Body):
    #...other TweetUserData model fields
    address: Optional[str]
    badges_minted: int
    quest_completed: int 
    total_badges_reward: int