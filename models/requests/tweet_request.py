

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
    bookmarkCount: int
    conversationId: str
    id: str
    hashtags: List[str]
    likes: int
    mentions: List[TweetMention]
    name: Optional[str]
    permanentUrl: Optional[str]
    photos: List[TweetPhoto]
    replies: int
    retweets: int
    text: str
    thread: List[str]
    urls: List[str]
    userId: str
    username: Optional[str]
    videos: List[TweetVideo]
    isQuoted: bool
    isReply: bool
    isRetweet: bool
    isPin: bool
    sensitiveContent: bool
    timeParsed: str
    timestamp: int
    html: str
    views: Optional[int] = None



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