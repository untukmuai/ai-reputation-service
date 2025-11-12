import os
from openai import AsyncOpenAI
from models.requests.dna_request import RequestDigitalDNA, RequestDigitalDNAImage
from utils.image_helper import get_average_hex_color
from utils.text_cleaner import emoji_to_codepoints
from google import genai
from google.genai.types import HarmCategory, HarmBlockThreshold
import orjson
import requests
from PIL import Image
from rembg import remove
from io import BytesIO
import base64
import logging

logger = logging.getLogger(__name__)

SAFETY_SETTINGS = [
    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
]

class DNAService:

    @staticmethod
    async def digital_dna_genai(payload: RequestDigitalDNA):
        try:
            max_tokens: int = 7000
            print('START digital_dna_genai')
            # print(payload.socmed_data)
            tw = payload.socmed_data.tweets
            labels = set(payload.unique_id)
            title = set(payload.title)
            texts = [
                {
                    "id": i.id,
                    "tweet": i.text,
                    "likes": i.likes,
                    "replies": i.replies,
                    "retweets": i.retweets,
                    "views": i.views or 0,  
                    "timeParsed": i.timeParsed,
                }
                for i in tw
            ]

            texts_dumps = orjson.dumps(texts)

            response_schema = {
                "type": "ARRAY",
                "description": "Given the user tweets, provide 10 categories(1-2 words) and its percentage sum up to 100% in total. Provide also example of the tweet, and 2 insights. If user's tweets less than 10, category created will be the same number like the number of tweets user's",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "category": {
                            "type": "STRING",
                            "description": "category name"
                        },
                        "description": {
                            "type": "STRING",
                            "description": "category description, 1 paragraph"
                        },
                        "percentage": {
                            "type": "INTEGER",
                            "description": "rough estimation of tweets related to this category, integer 0-100",
                            "minimum": 0,
                            "maximum": 100
                        },
                        "tweet": {
                            "type": "STRING",
                            "description": "a sample of tweet that falls to this category"
                        },
                        "likes": {
                            "type": "INTEGER",
                            "description": "total likes of that mention"
                        },
                        "replies": {
                            "type": "INTEGER",
                            "description": "total replies of that mention"
                        },
                        "retweets": {
                            "type": "INTEGER",
                            "description": "total retweets of that mention"
                        },
                        "views": {
                            "type": "INTEGER",
                            "description": "total views of that mention"
                        },
                        "timeParsed": {
                            "type": "STRING",
                            "description": "timeParsed of that mention"
                        },
                        "insights": {
                            "type": "ARRAY",
                            "description": "insights gathered from the tweet. Produce exactly two.",
                            "items": {
                                "type": "OBJECT",
                                    "properties": {
                                        "insight_title": {
                                            "type": "STRING",
                                            "description": "Insight title of the tweet"
                                        },
                                        "insight_description": {
                                            "type": "STRING",
                                            "description": "Insight description of the insight title"
                                        }
                                    },
                                "required": ["insight_title", "insight_description"]
                            }
                        }
                    },
                    "required": [
                        "category", 
                        "description", 
                        "percentage", 
                        "tweet", 
                        "likes", 
                        "replies", 
                        "retweets", 
                        "views", 
                        "timeParsed",
                        "insights"
                    ]
                }
            }
            client = genai.Client(api_key=os.getenv('GENAI_API_KEY'))
            tweet_count = len(texts)

            dna_generated_count = 10

            if tweet_count < 10 and tweet_count > 0:
                dna_generated_count = tweet_count
            
            # Strategy: Sample-based estimation for large datasets
            if tweet_count > 100:
                # Take a sample to estimate average tokens per tweet
                sample_size = min(50, len(texts))
                sample_texts = texts[:sample_size]
                sample_json = orjson.dumps(sample_texts)
                
                sample_resp = client.models.count_tokens(
                    model='gemini-2.0-flash-lite',
                    contents=sample_json
                )
                
                avg_tokens_per_tweet = sample_resp.total_tokens / sample_size
                estimated_max_tweets = int((max_tokens / avg_tokens_per_tweet) * 0.85)  # 85% safety margin
                
                # Apply estimation
                texts = texts[:estimated_max_tweets]
                texts_dumps = orjson.dumps(texts)
                
                # Single verification count
                token_resp = client.models.count_tokens(
                    model='gemini-2.0-flash-lite',
                    contents=texts_dumps
                )
                token_count = token_resp.total_tokens or 0
                
                # If still over (rare), do a single adjustment
                if token_count > max_tokens:
                    adjustment_ratio = max_tokens / token_count
                    texts = texts[:int(len(texts) * adjustment_ratio)]
                    texts_dumps = orjson.dumps(texts)
                    token_count = max_tokens  # Close enough
            else:
                # Small dataset, just count once
                texts_dumps = orjson.dumps(texts)
                token_resp = client.models.count_tokens(
                    model='gemini-2.0-flash-lite',
                    contents=texts_dumps
                )
                token_count = token_resp.total_tokens or 0
                
                if token_count > max_tokens:
                    ratio = max_tokens / token_count
                    texts = texts[:int(len(texts) * ratio * 0.9)]
                    texts_dumps = orjson.dumps(texts)
                    token_count = int(token_count * ratio * 0.9)

            current_tokens = token_count
            truncated_texts = texts
            
            # Process emojis after truncation (only for tweets we'll actually use)
            for tweet_obj in truncated_texts:
                tweet_obj['tweet'] = emoji_to_codepoints(tweet_obj['tweet'])
            
            texts_dumps = orjson.dumps(truncated_texts)
            
            # Pre-calculate title string and conditional prompt (Point 8: Fix conditional logic)
            thousand_prompt = "3. Please reuse the existing category only stated here, so don't create any new category"
            usual_prompt = "3. **Prioritize the following existing categories** (only invent a new one if none match semantically):"
            
            # Convert title set to sorted string for consistent ordering
            title_str = " ".join(sorted(title)) if isinstance(title, set) else " ".join(title)
            use_truncated = len(title_str) > 1000
            prompt_instruction = thousand_prompt if use_truncated else usual_prompt
            title_content = title_str[:1000] if use_truncated else title_str

            text_prompt = f"""Analyze these tweets and identify {dna_generated_count} distinct personality/content DNA traits.
            
            Tweets:
            {texts_dumps}

            Rules:
            1. Use DNA-style naming (1-2 words): "Tech Curiosity", "Crypto Enthusiasm", etc.
            2. Avoid semantic redundancy - reuse existing categories if meaning matches
            {prompt_instruction}
            {title_content}
            3. Identify {dna_generated_count} distict traits
            4. Percentages must total exactly 100%
            5. Each trait needs: category, description (1 paragraph), percentage, sample tweet with metrics, and 2 insights

            Don't repeat categories."""


            text_task = await client.aio.models.generate_content(
                model='gemini-2.0-flash-lite',
                contents=text_prompt,
                config={
                    'safety_settings': SAFETY_SETTINGS,
                    'response_mime_type': 'application/json',
                    'response_schema': response_schema,
                    "temperature": 0.9,        # Balanced creativity for structured output
                    "top_p": 0.98,            # Nearly full vocabulary access
                    "top_k": 64,              # Standard token sampling
                    "frequency_penalty": 0.6,  # Moderate anti-repetition
                    "presence_penalty": 0.4,   # Encourage topic diversity
                }
            )

            # Point 10: Fix response text cleaning with proper methods
            response_text = text_task.text
            response_text = (
                response_text
                .strip()
                .removeprefix('```json')
                .removeprefix('```')
                .removesuffix('```')
                .strip()
            )
            response_text_dict = orjson.loads(response_text)

            # Point 9: Use dictionary for O(1) duplicate handling instead of O(n²) linear search
            dna_dict = {}
            new_dna = []

            for val in response_text_dict:
                # Normalize to unique_id early for case-insensitive matching
                category_name = val['category']
                unique_id = "_".join(category_name.lower().split())
                
                # Normalize percentage early
                percentage = val.get('percentage', 0)
                if isinstance(percentage, str):
                    percentage = int(percentage.rstrip('%')) if '%' in percentage else int(percentage)
                else:
                    percentage = int(percentage)
                
                # Check for duplicate using unique_id (O(1) lookup)
                if unique_id in dna_dict:
                    dna_dict[unique_id]['percentage'] += percentage
                    continue
                
                # Build new entry
                entry = {
                    "unique_id": unique_id,
                    "title": category_name,
                    "description": val['description'],
                    "percentage": percentage,
                    "tweet_mention": val["tweet"],
                    "likes": int(val["likes"]),
                    "replies": int(val["replies"]),
                    "retweets": int(val["retweets"]),
                    "views": int(val["views"]),
                    "time": val["timeParsed"],
                    "ai_insight": val["insights"]
                }
                
                dna_dict[unique_id] = entry
                
                # Track new categories
                if unique_id not in labels:
                    new_dna.append({
                        "unique_id": unique_id,
                        "title": category_name,
                        "description": val['description'],
                    })

            # Convert dict to list for response
            dna = list(dna_dict.values())

            return {
                "original_token": token_count,
                "cut_token" : current_tokens,
                "free_tweets": len(truncated_texts),
                "dna" : dna,
                "new_dna": new_dna
            }
        except Exception as e:
            logger.exception("digital_dna_genai_err: %s", e)
            raise e

    @staticmethod
    async def generate_dna_image(payload: RequestDigitalDNAImage) -> str:
        try:
            prompt = f"""You are a visual design AI trained to generate stylized gamified badge icons. Your task is to create a descriptive visual concept for a badge icon based on the input title, using the style described below.

            **Style Guide:**
            - Modern, 3D-styled hexagon or shield-shaped badge
            - Glowing gradients and high contrast colors
            - Smooth shadows and lighting effects
            - Symbolic, fantasy-style icon in the center (e.g. wand, wings, vault, camera)
            - Progress bar or visual indicator optional
            - Similar to mobile RPG/UI badges or NFT gamification

            **Input Title:** {payload.title}

            **Instructions:**
            1. Identify a metaphor or symbol to represent the input (e.g., “wand” for DeFi = magical finance).
            2. Suggest a color palette that matches the concept (e.g., green for finance, blue for trust).
            3. Describe the icon’s shape and what’s in the center (e.g., golden shield with a glowing lamp).
            4. Match the polished, glowing 3D style with fantasy or futuristic elements.
            5. Output a concise description suitable for an image generation AI prompt.

            **Example Output for “Personal Finance Advisor”:**  
            A polished blue hexagonal badge with a glowing golden winged lamp in the center, symbolizing guidance and trust. Smooth gradient from dark to electric blue, accented with sparkles. The icon has a soft shadow, glowing edges, and resembles a gamified digital achievement badge."""
            client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

            response = await client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )

            response = requests.get(response.data[0].url)
            response.raise_for_status()

            input_image = Image.open(BytesIO(response.content))
            nobg_image = remove(input_image)

            average_hex = get_average_hex_color(nobg_image)

            buffer = BytesIO()
            nobg_image.save(buffer, format="PNG")
            buffer.seek(0)
            image_b64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            image = {
                "image_b64": image_b64,
                "background_hex": average_hex
            }

            return image
        except Exception as e:
            logger.exception("generate_dna_image_err: %s", e)
            raise