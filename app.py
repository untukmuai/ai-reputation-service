from pydantic import ValidationError
from robyn import Request, Response, Robyn
from controllers.dna_controller import DNAController
from controllers.identifi_controller import IdentifiController
from controllers.persona_controller import PersonaController
from models.requests.identifi_request import RequestIdentifiScore
from models.responses.base_response import BaseResponse, ErrorResponse
from services.identifi_service import IdentifiScore
from utils.libs_loader import libs_loader
import orjson

app = Robyn(__file__)

print("Initializing AI Rep Service")
libs_loader.load_all()

DNAController(app)
PersonaController(app)
IdentifiController(app)

if __name__ == "__main__":
    app.start(host="0.0.0.0", port=8080)
