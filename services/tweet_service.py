import os
from google import genai
from google.genai.types import HarmBlockThreshold, HarmCategory
from models.requests.tweet_request import RequestAnalyzeTweet
import orjson
import logging


logger = logging.getLogger(__name__)

SAFETY_SETTINGS = [
    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
]

class TweetService:

    @staticmethod
    async def analyze_single_tweet(payload: RequestAnalyzeTweet):
        try:
            text_prompt = f"""
            Analyze the following tweet for sentiment, credibility, and other insights. Provide a detailed analysis in JSON format with the following structure:

            {{
                "sentiment": {{
                    "overall": "positive|neutral|negative",
                    "score": 0-100,
                    "description": "Detailed sentiment analysis"
                }},
                "credibility": {{
                    "score": 0-100,
                    "level": "high|medium|low",
                    "factors": ["factor1", "factor2"],
                    "description": "Credibility assessment"
                }},
                "content_analysis": {{
                    "topics": ["topic1", "topic2"],
                    "tone": "formal|informal|casual|professional",
                    "language_quality": "excellent|good|fair|poor"
                }},
                "potential_issues": ["issue1", "issue2"],
                "recommendations": ["recommendation1", "recommendation2"]
            }}

            Tweet: {payload.tweet_text}
            Author: {payload.author}

            Please provide a comprehensive analysis focusing on:
            1. Sentiment analysis (positive, neutral, negative)
            2. Credibility assessment based on content quality, fact-checking potential, and language use
            3. Content analysis including topics and tone
            4. Potential issues like misinformation, bias, or inappropriate content
            5. Recommendations for readers
            """
            client = genai.Client(api_key=os.getenv('GENAI_API_KEY'))
            text_task = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents = text_prompt,
                config={
                    'safety_settings': SAFETY_SETTINGS,
                    'response_mime_type': 'application/json',
                }
            )

            response_text = text_task.text
            response_text = response_text.strip('```json\n')
            response_text = response_text.strip('\n``` \n')
            response_text_dict = orjson.loads(response_text)

            return {
                "tweet_text" : payload.tweet_text,
                "author": payload.author,
                **response_text_dict
            }
        except Exception as e:
            logger.exception("analyze_single_tweet_err: %s", e)
            raise