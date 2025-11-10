from models.requests.persona_request import RequestSortingHat
from models.responses.base_response import BaseResponse, ErrorResponse
from utils.libs_loader import libs_loader
import orjson
from openai import AsyncOpenAI
import os

from enum import Enum

class PersonaChain(Enum):
    BNB='bnb'
    SOMNIA='somnia'

class PersonaService:
    """Service to handle user persona classification."""

    @staticmethod
    async def get_persona(payload: RequestSortingHat, chain: PersonaChain):
        try:
            texts_dna = orjson.dumps(payload.digital_dna)
            client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

            if chain == PersonaChain.BNB:
                persona_config = libs_loader.get_raw('persona_bnb')
            elif chain == PersonaChain.SOMNIA:
                persona_config = libs_loader.get_raw('persona_somnia')

            if not payload.old_persona:
                prompt = f"""
                        Based on this person's digital dna or traits:
                        {texts_dna}

                        Decide persona and the tier based on this hint:
                        {persona_config}

                        With this format:
                        {{
                            "persona":<the persona>,
                            "tier":<1,2,3 in integer>
                        }}
                        """
            else:
                prompt = f"""
                        Based on this person's digital dna or traits:
                        {texts_dna}

                        And his/her old persona and tier:
                        Old Persona: {payload.old_persona}
                        Tier: {payload.old_tier}

                        Decide new persona and the tier based on this hint:
                        {persona_config}

                        With this format:
                        {{
                            "persona":<the persona>,
                            "tier":<1,2,3 in integer>,
                            "reasons_for_change":<narrative reason for change, 1-2 sentence.>
                        }}
                        """


            response = await client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to create output in json format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            seed=42
            )

            response_text = response.choices[0].message.content
            description = orjson.loads(response_text)
            description["reasons_for_change"] = description.get("reasons_for_change")
            description['tier'] = int(description['tier'])
            return description
        except Exception as e:
            raise e
    