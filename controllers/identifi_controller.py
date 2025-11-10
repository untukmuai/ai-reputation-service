from robyn import Request, Robyn, Response
from typing import Dict, Any
import orjson

from services.identifi_service import IdentifiScore
from models.requests.identifi_request import RequestIdentifiScore
from models.responses.base_response import BaseResponse, ErrorResponse


class IdentifiController:
    def __init__(self, app: Robyn):
        self.app = app
        self._register_routes()
    
    def _register_routes(self):
        """Register all routes for this controller"""
        self.app.post("/api/identifi/log", 
        openapi_tags=["Identifi Score"], 
        openapi_name="Get identifi score logarithmic")(self.get_identifi_score_log)

    async def get_identifi_score_log(self, request: Request, body: RequestIdentifiScore) -> Response:
        try:
            # payload = orjson.loads(request.body)
            payload_body = orjson.loads(body)
            validated_payload =  RequestIdentifiScore(**payload_body)
            result = await IdentifiScore.calculate_identifi_log(validated_payload)
            
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