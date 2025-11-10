from robyn import Request, Robyn, Response
from typing import Dict, Any
import orjson

from services.persona_service import PersonaService, PersonaChain
from models.requests.persona_request import RequestSortingHat
from models.responses.base_response import BaseResponse, ErrorResponse

class PersonaController:

    def __init__(self, app: Robyn):
        self.app = app
        self._register_routes()
    
    def _register_routes(self):
        """Register all routes for this controller"""
        self.app.post("/api/persona/bnb", openapi_tags=["Persona"], openapi_name="Get BNB Persona")(self.get_persona_bnb)
        self.app.post("/api/persona/somnia", openapi_tags=["Persona"], openapi_name="Get Somnia Persona")(self.get_persona_somnia)

    async def get_persona_somnia(self, request: Request, body: RequestSortingHat) -> Response:
        try:
            payload = orjson.loads(request.body)
            validated_payload =  RequestSortingHat(**payload)
            result = await PersonaService.get_persona(validated_payload, PersonaChain.SOMNIA)
            
            success_response = BaseResponse(
                success=True,
                message="Data generated successfully",
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

    async def get_persona_bnb(self, request: Request, body: RequestSortingHat) -> Response:
        try:
            payload = orjson.loads(request.body)
            validated_payload =  RequestSortingHat(**payload)
            result = await PersonaService.get_persona(validated_payload, PersonaChain.BNB)
            
            success_response = BaseResponse(
                success=True,
                message="Data generated successfully",
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

    