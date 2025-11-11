from robyn import Request, Robyn, Response
from typing import Dict, Any
import orjson

from services.dna_service import DNAService
from models.requests.dna_request import RequestDigitalDNA, RequestDigitalDNAImage
from models.responses.base_response import BaseResponse, ErrorResponse


class DNAController:
    """Controller for handling Digital DNA API requests"""
    
    def __init__(self, app: Robyn):
        self.app = app
        self._register_routes()
    
    def _register_routes(self):
        """Register all routes for this controller"""
        self.app.post("/api/dna/generate", openapi_tags=["DNA"], openapi_name="Get Digital DNA")(self.generate_digital_dna)
        self.app.post("/api/dna/image", openapi_tags=["DNA"], openapi_name="Generate DNA Image")(self.generate_dna_image)
    
    async def generate_digital_dna(self, request: Request, body: RequestDigitalDNA) -> Response:
        """
        Handle POST /api/dna/generate endpoint
        
        Analyze tweets and generate digital DNA categories with insights.
        """
        try:
            payload = orjson.loads(request.body)
            validated_payload =  RequestDigitalDNA(**payload)
            result = await DNAService.digital_dna_genai(validated_payload)
            
            success_response = BaseResponse(
                success=True,
                message="OK",
                data=result
            )
            return Response(
                status_code=200,
                headers={"Content-Type": "application/json"},
                description=orjson.dumps(success_response.model_dump())
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
                description=orjson.dumps(error_response.model_dump())
            )

    async def generate_dna_image(self, request: Request, body: RequestDigitalDNAImage) -> Response:
        try:
            payload = orjson.loads(request.body)
            result = await DNAService.generate_dna_image(RequestDigitalDNAImage(**payload))
            response = BaseResponse(success=True, message="OK", data=result)
            return Response(
                status_code=200, 
                headers={"Content-Type": "application/json"},
                description=orjson.dumps(response.model_dump())
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
                description=orjson.dumps(error_response.model_dump())
            )