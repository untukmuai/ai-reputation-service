

import orjson
from robyn import Robyn, Request, Response

from models.requests.tweet_request import RequestAnalyzeTweet
from models.responses.base_response import BaseResponse, ErrorResponse
from services.tweet_service import TweetService


class TweetController:
    def __init__(self, app: Robyn):
        self.app = app
        self._register_routes()
    
    def _register_routes(self):
        """Register all routes for this controller"""
        self.app.post("/api/tweet/analyze", openapi_tags=["Tweet"], openapi_name="Analyze single tweet")(self.analyze_tweet)

    async def analyze_tweet(self, request: Request, body: RequestAnalyzeTweet) -> Response:
        try:
            payload = orjson.loads(request.body)
            result = await TweetService.analyze_single_tweet(RequestAnalyzeTweet(**payload))
            
            success_response = BaseResponse(
                success=True,
                message="OK",
                data=result
            )
            return Response(
                status_code=200,
                headers={"Content-Type": "application/json"},
                description=orjson.dumps(success_response.dict())
            )  
        except Exception as e:
            error_response = ErrorResponse(
                success=False,
                message="Internal server error",
                error_code="INTERNAL_ERROR",
                details={"error": str(e)}
            )
            return Response(
                status_code=500,
                headers={"Content-Type": "application/json"},
                description=orjson.dumps(error_response.dict())
            )