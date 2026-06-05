

from typing import Optional, List
from pydantic import BaseModel
from robyn.types import Body

class TweetPhoto(BaseModel, Body):
    id: str
    url: str

class TweetVideo(BaseModel, Body):
    id: str
    preview: str
    url: str

class TweetMention(BaseModel, Body):
    id: str
    username: str
    name: str


class Tweets(BaseModel, Body):
    id: str
    hashtags: List[str]
    likes: int
    mentions: List[TweetMention]
    name: Optional[str]
    photos: List[TweetPhoto]
    replies: int
    retweets: int
    views: Optional[int] = None
    text: str
    userId: str
    username: Optional[str]
    videos: List[TweetVideo]
    isQuoted: bool
    isReply: bool
    isRetweet: bool
    timestamp: int
    postedAt: str
    quotes: int
    # bookmarkCount: int
    # conversationId: str
    # permanentUrl: Optional[str]
    # thread: List[str]
    # urls: List[str]
    # isPin: bool
    # sensitiveContent: bool
    # timeParsed: str
    # html: str




class TweetPublicMetrics(BaseModel, Body):
    followers_count: int
    following_count: int
    tweet_count: int
    listed_count: Optional[int]

class TweetUserData(BaseModel, Body):
    username: str
    name: str
    profile_picture_url: Optional[str]
    location: Optional[str]
    public_metrics: TweetPublicMetrics
    tweets: List[Tweets]

class RequestAnalyzeTweet(BaseModel, Body):
    tweet_text: str
    author: str