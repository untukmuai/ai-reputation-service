
from typing import List, Optional
from pydantic import BaseModel
from robyn.types import Body
from typing_extensions import Literal

from models.requests.tweet_request import TweetUserData

class BaseFeedbackScorePayload(Body):
    followers: int
    vote: Literal["up", "down"]
    twitter_account_age_days: int
    quality_score: str


class RequestIdentifiScore(TweetUserData, Body):
    #...other TweetUserData model fields
    address: Optional[str]
    badges_minted: int
    quest_completed: int 
    total_badges_reward: int

class RequestIdentifiScoreV2(TweetUserData, Body):
    #...other TweetUserData model fields
    voters: Optional[List[BaseFeedbackScorePayload]]
    address: Optional[str]
    badges_minted: int
    quest_completed: int 
    total_badges_reward: int