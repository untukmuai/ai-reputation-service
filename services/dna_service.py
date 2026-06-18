import os
import copy
from collections import Counter
from openai import AsyncOpenAI
from models.requests.dna_request import RequestDigitalDNA, RequestDigitalDNAImage
from utils.image_helper import get_average_hex_color
from utils.text_cleaner import emoji_to_codepoints
from google import genai
from google.genai.types import HarmCategory, HarmBlockThreshold
from sentence_transformers import util
from services.identifi_service import embedder
import orjson
from PIL import Image
from rembg import remove
from io import BytesIO
import base64
import logging

logger = logging.getLogger(__name__)

DNA_TINY_THRESHOLD = int(os.getenv("DNA_TINY_THRESHOLD", "50"))
DNA_CAP_THRESHOLD = int(os.getenv("DNA_CAP_THRESHOLD", "1000"))
DNA_SHORTLIST_SIZE = int(os.getenv("DNA_SHORTLIST_SIZE", "30"))
DNA_CLASSIFICATION_SHORTLIST_SIZE = int(os.getenv("DNA_CLASSIFICATION_SHORTLIST_SIZE", "20"))
DNA_SIMILARITY_THRESHOLD = float(os.getenv("DNA_SIMILARITY_THRESHOLD", "0.75"))
DNA_DISCOVERY_TEMPERATURE = float(os.getenv("DNA_DISCOVERY_TEMPERATURE", "0.3"))
DNA_TINY_TEMPERATURE = float(os.getenv("DNA_TINY_TEMPERATURE", "0.4"))
DNA_CLASSIFICATION_TEMPERATURE = float(os.getenv("DNA_CLASSIFICATION_TEMPERATURE", "0.3"))
DNA_LLM_SEED = int(os.getenv("DNA_LLM_SEED", "42"))
DNA_LLM_TOP_P = float(os.getenv("DNA_LLM_TOP_P", "0.9"))
DNA_LLM_TOP_K = int(os.getenv("DNA_LLM_TOP_K", "40"))
DNA_UNMATCHED_THRESHOLD = float(os.getenv("DNA_UNMATCHED_THRESHOLD", "0.70"))
DNA_UNMATCHED_MIN_TWEETS = int(os.getenv("DNA_UNMATCHED_MIN_TWEETS", "2"))
DNA_NEW_DNA_MAX_CLUSTERS = int(os.getenv("DNA_NEW_DNA_MAX_CLUSTERS", "3"))
DNA_CLUSTER_THRESHOLD = float(os.getenv("DNA_CLUSTER_THRESHOLD", "0.75"))
DNA_NEW_DNA_NAMING_TEMPERATURE = float(os.getenv("DNA_NEW_DNA_NAMING_TEMPERATURE", "0.2"))

NEW_DNA_NAMING_SCHEMA = {
    "type": "ARRAY",
    "description": "Proposed new DNA categories for unmatched tweet clusters.",
    "items": {
        "type": "OBJECT",
        "properties": {
            "cluster_id": {
                "type": "INTEGER",
                "description": "cluster index from the input"
            },
            "title": {
                "type": "STRING",
                "description": "new DNA category name, 1-2 words"
            },
            "description": {
                "type": "STRING",
                "description": "one paragraph category description"
            }
        },
        "required": ["cluster_id", "title", "description"]
    }
}

INSIGHT_REGENERATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "insights": {
            "type": "ARRAY",
            "description": "Exactly two insights about the tweet for the given category.",
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
    "required": ["insights"]
}

DNA_INSIGHT_REGEN_TEMPERATURE = float(os.getenv("DNA_INSIGHT_REGEN_TEMPERATURE", "0.2"))

SAFETY_SETTINGS = [
    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
]


class DNAService:

    @staticmethod
    def _get_base_response_schema():
        return {
            "type": "ARRAY",
            "description": "Given the user tweets, provide categories and insights. Percentages are recalculated server-side.",
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
                        "description": "placeholder percentage (recalculated server-side from tweet counts)",
                        "minimum": 0,
                        "maximum": 100
                    },
                    "tweet_id": {
                        "type": "STRING",
                        "description": "Exact id of the sample tweet from the input tweets array"
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
                    "tweet_id",
                    "insights"
                ]
            }
        }

    @staticmethod
    def _normalize_unique_id(category_name: str) -> str:
        return "_".join(category_name.lower().split())

    @staticmethod
    def _build_title_to_uid_map(titles: list, unique_ids: list) -> dict:
        mapping = {}
        for title, uid in zip(titles, unique_ids):
            mapping[title] = uid
            mapping[title.lower()] = uid
        return mapping

    @staticmethod
    def _build_uid_to_title_map(titles: list, unique_ids: list) -> dict:
        return dict(zip(unique_ids, titles))

    @staticmethod
    def _resolve_unique_id(category_name: str, title_to_uid: dict) -> str:
        return (
            title_to_uid.get(category_name)
            or title_to_uid.get(category_name.lower())
            or DNAService._normalize_unique_id(category_name)
        )

    @staticmethod
    def _build_label_embeddings(label_titles: list):
        if not label_titles:
            return None
        return embedder.encode(
            label_titles, normalize_embeddings=True, show_progress_bar=False
        )

    @staticmethod
    def _build_shortlist(label_titles: list, truncated_texts: list, top_k: int):
        if not label_titles:
            return [], None

        label_embeddings = DNAService._build_label_embeddings(label_titles)

        if not truncated_texts:
            top_k = min(top_k, len(label_titles))
            shortlist_titles = label_titles[:top_k]
            return shortlist_titles, label_embeddings

        sample_tweets = [t["tweet"] for t in truncated_texts[:30]]
        tweet_embeddings = embedder.encode(
            sample_tweets, normalize_embeddings=True, show_progress_bar=False
        )
        query_vec = tweet_embeddings.mean(axis=0, keepdims=True)

        sims = util.cos_sim(query_vec, label_embeddings)[0].cpu().numpy()
        top_k = min(top_k, len(label_titles))
        top_indices = sims.argsort()[-top_k:][::-1]
        shortlist_titles = [label_titles[i] for i in top_indices]

        return shortlist_titles, label_embeddings

    @staticmethod
    def _build_classification_schema(base_schema: dict, shortlist_titles: list) -> dict:
        schema = copy.deepcopy(base_schema)
        schema["items"]["properties"]["category"] = {
            "type": "STRING",
            "enum": shortlist_titles,
            "description": "category name - must be one of the allowed categories"
        }
        return schema

    @staticmethod
    def _build_active_schema(base_schema: dict, shortlist_titles: list, tweet_ids: list) -> dict:
        schema = DNAService._build_classification_schema(base_schema, shortlist_titles)
        schema["items"]["properties"]["tweet_id"] = {
            "type": "STRING",
            "enum": tweet_ids,
            "description": "Exact id of the sample tweet from the input tweets array"
        }
        return schema

    @staticmethod
    def _build_tweet_by_id(truncated_texts: list) -> dict:
        return {str(tweet["id"]): tweet for tweet in truncated_texts}

    @staticmethod
    def _apply_entry_from_tweet(entry: dict, tweet: dict):
        entry["tweet_id"] = str(tweet["id"])
        entry["tweet_mention"] = tweet["tweet"]
        entry["likes"] = int(tweet["likes"])
        entry["replies"] = int(tweet["replies"])
        entry["retweets"] = int(tweet["retweets"])
        entry["views"] = int(tweet["views"])
        entry["time"] = tweet["postedAt"]

    @staticmethod
    def _resolve_canonical_label(
        entry: dict,
        labels: set,
        label_titles: list,
        label_embeddings,
        title_to_uid: dict,
        uid_to_title: dict,
        threshold: float,
    ):
        category_name = entry["title"]
        unique_id = entry["unique_id"]

        if category_name in title_to_uid:
            uid = title_to_uid[category_name]
            return uid, uid_to_title.get(uid, category_name), True

        if unique_id in labels:
            return unique_id, uid_to_title.get(unique_id, category_name), True

        if label_embeddings is not None and label_titles:
            candidate_embed = embedder.encode(
                [category_name], normalize_embeddings=True, show_progress_bar=False
            )
            sims = util.cos_sim(candidate_embed, label_embeddings)[0].cpu().numpy()
            max_sim = float(sims.max())
            nearest_idx = int(sims.argmax())

            if max_sim >= threshold:
                nearest_title = label_titles[nearest_idx]
                uid = title_to_uid.get(nearest_title) or DNAService._normalize_unique_id(nearest_title)
                return uid, nearest_title, True

        return unique_id, category_name, unique_id in labels

    @staticmethod
    def _canonicalize_dna(
        dna_dict: dict,
        labels: set,
        label_titles: list,
        label_embeddings,
        mode: str,
        threshold: float,
        title_to_uid: dict,
        uid_to_title: dict,
    ):
        processed = {}

        for entry in dna_dict.values():
            unique_id, title, _ = DNAService._resolve_canonical_label(
                entry=entry,
                labels=labels,
                label_titles=label_titles,
                label_embeddings=label_embeddings,
                title_to_uid=title_to_uid,
                uid_to_title=uid_to_title,
                threshold=threshold,
            )

            if mode == "classification" and unique_id not in labels:
                continue

            canonical_entry = entry.copy()
            canonical_entry["unique_id"] = unique_id
            canonical_entry["title"] = title

            if unique_id in processed:
                processed[unique_id]["percentage"] += canonical_entry["percentage"]
                continue

            processed[unique_id] = canonical_entry

        return list(processed.values()), []

    @staticmethod
    def _find_unmatched_tweets(truncated_texts: list, label_embeddings, threshold: float) -> list:
        if not truncated_texts or label_embeddings is None:
            return []

        tweet_texts = [t["tweet"] for t in truncated_texts]
        tweet_embs = embedder.encode(
            tweet_texts, normalize_embeddings=True, show_progress_bar=False
        )
        sims = util.cos_sim(tweet_embs, label_embeddings).cpu().numpy()
        max_sims = sims.max(axis=1)

        return [
            truncated_texts[idx]
            for idx, max_sim in enumerate(max_sims)
            if float(max_sim) < threshold
        ]

    @staticmethod
    def _cluster_unmatched_tweets(unmatched_tweets: list, max_clusters: int) -> list:
        if not unmatched_tweets:
            return []

        texts = [t["tweet"] for t in unmatched_tweets]
        embs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        assigned = set()
        clusters = []

        for i in range(len(unmatched_tweets)):
            if i in assigned:
                continue

            cluster = [unmatched_tweets[i]]
            assigned.add(i)

            for j in range(i + 1, len(unmatched_tweets)):
                if j in assigned:
                    continue
                sim = float(util.cos_sim(embs[i : i + 1], embs[j : j + 1]).item())
                if sim >= DNA_CLUSTER_THRESHOLD:
                    cluster.append(unmatched_tweets[j])
                    assigned.add(j)

            clusters.append(cluster)
            if len(clusters) >= max_clusters:
                break

        clusters.sort(key=len, reverse=True)
        return clusters[:max_clusters]

    @staticmethod
    def _filter_proposed_new_dna(
        proposals: list,
        labels: set,
        label_titles: list,
        label_embeddings,
        threshold: float,
    ) -> list:
        if not proposals:
            return []

        filtered = []
        seen_uids = set()

        for proposal in proposals:
            title = proposal["title"]
            unique_id = DNAService._normalize_unique_id(title)

            if unique_id in labels or unique_id in seen_uids:
                continue

            if label_embeddings is not None and label_titles:
                candidate_embed = embedder.encode(
                    [title], normalize_embeddings=True, show_progress_bar=False
                )
                max_sim = float(
                    util.cos_sim(candidate_embed, label_embeddings).max().item()
                )
                if max_sim >= threshold:
                    continue

            seen_uids.add(unique_id)
            filtered.append({
                "unique_id": unique_id,
                "title": title,
                "description": proposal["description"],
            })

        return filtered

    @staticmethod
    async def _propose_new_dna_from_unmatched(
        client,
        unmatched_tweets: list,
        labels: set,
        label_titles: list,
        label_embeddings,
    ) -> list:
        clusters = DNAService._cluster_unmatched_tweets(
            unmatched_tweets, DNA_NEW_DNA_MAX_CLUSTERS
        )
        if not clusters:
            return []

        cluster_payload = [
            {
                "cluster_id": idx,
                "tweet_count": len(cluster),
                "sample_tweets": [tweet["tweet"] for tweet in cluster[:5]],
            }
            for idx, cluster in enumerate(clusters)
        ]

        existing_hint = ", ".join(label_titles[:80])
        naming_prompt = f"""These tweet clusters do not match any existing DNA category closely enough.
        Existing categories (do NOT duplicate or rephrase these):
        {existing_hint}

        For each cluster below, propose exactly ONE new DNA category:
        - Use DNA-style naming (1-2 words), e.g. "Tech Curiosity", "Crypto Enthusiasm"
        - Provide a one-paragraph description
        - Return one proposal per cluster_id

        Clusters:
        {orjson.dumps(cluster_payload).decode()}"""

        naming_config = DNAService._build_llm_config(
            DNA_NEW_DNA_NAMING_TEMPERATURE, NEW_DNA_NAMING_SCHEMA
        )

        try:
            naming_task = await client.aio.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=naming_prompt,
                config=naming_config,
            )
        except Exception as seed_err:
            if "seed" in str(seed_err).lower():
                logger.warning(
                    "New DNA naming seed not supported, retrying without seed: %s", seed_err
                )
                naming_config.pop("seed", None)
                naming_task = await client.aio.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=naming_prompt,
                    config=naming_config,
                )
            else:
                raise

        response_text = naming_task.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        proposals = orjson.loads(response_text)

        return DNAService._filter_proposed_new_dna(
            proposals=proposals,
            labels=labels,
            label_titles=label_titles,
            label_embeddings=label_embeddings,
            threshold=DNA_SIMILARITY_THRESHOLD,
        )

    @staticmethod
    async def _discover_new_dna_hybrid(
        client,
        mode: str,
        truncated_texts: list,
        labels: set,
        label_titles: list,
        label_embeddings,
    ) -> list:
        if mode not in ("tiny", "discovery"):
            return []

        unmatched = DNAService._find_unmatched_tweets(
            truncated_texts, label_embeddings, DNA_UNMATCHED_THRESHOLD
        )
        if len(unmatched) < DNA_UNMATCHED_MIN_TWEETS:
            logger.info(
                "new_dna discovery skipped: %s unmatched tweets (min %s)",
                len(unmatched),
                DNA_UNMATCHED_MIN_TWEETS,
            )
            return []

        logger.info("new_dna discovery: %s unmatched tweets", len(unmatched))
        return await DNAService._propose_new_dna_from_unmatched(
            client=client,
            unmatched_tweets=unmatched,
            labels=labels,
            label_titles=label_titles,
            label_embeddings=label_embeddings,
        )

    @staticmethod
    def _pick_sample_tweet_index(
        category_idx: int,
        assignments,
        sims,
        truncated_texts: list,
    ) -> int:
        non_empty_indices = [
            i for i, tweet in enumerate(truncated_texts) if tweet["tweet"].strip()
        ]
        if not non_empty_indices:
            return int(sims[:, category_idx].argmax())

        assigned_indices = [
            i for i, assigned in enumerate(assignments) if assigned == category_idx
        ]
        non_empty_assigned = [
            i for i in assigned_indices if truncated_texts[i]["tweet"].strip()
        ]

        if non_empty_assigned:
            return max(non_empty_assigned, key=lambda i: sims[i, category_idx])

        return max(non_empty_indices, key=lambda i: sims[i, category_idx])

    @staticmethod
    def _compute_category_tweet_sims(dna_list: list, truncated_texts: list):
        scorable_texts = [
            (idx, tweet)
            for idx, tweet in enumerate(truncated_texts)
            if tweet["tweet"].strip()
        ]
        if not scorable_texts or not dna_list:
            return None

        tweet_indices = [idx for idx, _ in scorable_texts]
        tweet_texts = [tweet["tweet"] for _, tweet in scorable_texts]
        category_titles = [entry["title"] for entry in dna_list]

        tweet_embs = embedder.encode(
            tweet_texts, normalize_embeddings=True, show_progress_bar=False
        )
        category_embs = embedder.encode(
            category_titles, normalize_embeddings=True, show_progress_bar=False
        )

        sims = util.cos_sim(tweet_embs, category_embs).cpu().numpy()
        return {
            "sims": sims,
            "tweet_indices": tweet_indices,
            "scorable_texts": [tweet for _, tweet in scorable_texts],
            "total": len(scorable_texts),
        }

    @staticmethod
    def _apply_tweet_percentages(dna_list: list, sims_data: dict):
        if not dna_list or not sims_data:
            return

        sims = sims_data["sims"]
        assignments = sims.argmax(axis=1)
        counts = Counter(int(i) for i in assignments)
        total = sims_data["total"]

        percentages = []
        for idx in range(len(dna_list)):
            pct = round(counts.get(idx, 0) / total * 100)
            percentages.append(pct)

        drift = 100 - sum(percentages)
        if drift != 0 and percentages:
            adjust_idx = max(range(len(dna_list)), key=lambda i: counts.get(i, 0))
            percentages[adjust_idx] += drift

        for idx, entry in enumerate(dna_list):
            entry["percentage"] = percentages[idx]

    @staticmethod
    async def _generate_content(client, prompt: str, config: dict):
        try:
            return await client.aio.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config=config,
            )
        except Exception as seed_err:
            if "seed" in str(seed_err).lower():
                logger.warning("LLM seed not supported, retrying without seed: %s", seed_err)
                config = config.copy()
                config.pop("seed", None)
                return await client.aio.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt,
                    config=config,
                )
            raise

    @staticmethod
    async def _regenerate_insights(client, category_title: str, tweet_text: str) -> list:
        prompt = f"""Generate exactly 2 insights for the tweet below under the DNA category "{category_title}".
        Insights must directly reference the tweet content.

        Tweet:
        {tweet_text}"""

        config = DNAService._build_llm_config(
            DNA_INSIGHT_REGEN_TEMPERATURE, INSIGHT_REGENERATION_SCHEMA
        )
        task = await DNAService._generate_content_with_seed_fallback(client, prompt, config)
        response_text = (
            task.text.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        payload = orjson.loads(response_text)
        return payload["insights"]

    @staticmethod
    async def _resolve_tweet_samples(
        client,
        dna_list: list,
        truncated_texts: list,
        tweet_by_id: dict,
        sims_data: dict,
    ):
        if not dna_list:
            return

        for idx, entry in enumerate(dna_list):
            tweet_id = str(entry.get("tweet_id", "")).strip()
            mapped_tweet = tweet_by_id.get(tweet_id)

            if mapped_tweet and mapped_tweet["tweet"].strip():
                DNAService._apply_entry_from_tweet(entry, mapped_tweet)
                continue

            if not sims_data:
                continue

            best_local_idx = DNAService._pick_sample_tweet_index(
                category_idx=idx,
                assignments=sims_data["sims"].argmax(axis=1),
                sims=sims_data["sims"],
                truncated_texts=sims_data["scorable_texts"],
            )
            best_tweet_idx = sims_data["tweet_indices"][best_local_idx]
            fallback_tweet = truncated_texts[best_tweet_idx]
            DNAService._apply_entry_from_tweet(entry, fallback_tweet)

            if fallback_tweet["tweet"].strip():
                entry["ai_insight"] = await DNAService._regenerate_insights(
                    client,
                    entry["title"],
                    fallback_tweet["tweet"],
                )
                logger.info(
                    "Regenerated insights for %s using fallback tweet id %s",
                    entry["unique_id"],
                    fallback_tweet["id"],
                )

    @staticmethod
    def _parse_llm_response(response_text_dict: list, title_to_uid: dict, tweet_by_id: dict) -> dict:
        dna_dict = {}

        for val in response_text_dict:
            category_name = val["category"]
            unique_id = DNAService._resolve_unique_id(category_name, title_to_uid)

            percentage = val.get("percentage", 0)
            if isinstance(percentage, str):
                percentage = int(percentage.rstrip("%")) if "%" in percentage else int(percentage)
            else:
                percentage = int(percentage)

            if unique_id in dna_dict:
                dna_dict[unique_id]["percentage"] += percentage
                continue

            tweet_id = str(val.get("tweet_id", "")).strip()
            mapped_tweet = tweet_by_id.get(tweet_id)

            entry = {
                "unique_id": unique_id,
                "title": category_name,
                "description": val["description"],
                "percentage": percentage,
                "tweet_id": tweet_id,
                "tweet_mention": "",
                "likes": 0,
                "replies": 0,
                "retweets": 0,
                "views": 0,
                "time": "",
                "ai_insight": val["insights"],
            }

            if mapped_tweet:
                DNAService._apply_entry_from_tweet(entry, mapped_tweet)

            dna_dict[unique_id] = entry

        return dna_dict

    @staticmethod
    def _build_llm_config(temperature: float, response_schema: dict) -> dict:
        config = {
            "safety_settings": SAFETY_SETTINGS,
            "response_mime_type": "application/json",
            "response_schema": response_schema,
            "temperature": temperature,
            "top_p": DNA_LLM_TOP_P,
            "top_k": DNA_LLM_TOP_K,
            "seed": DNA_LLM_SEED,
        }
        return config

    @staticmethod
    async def digital_dna_genai(payload: RequestDigitalDNA):
        try:
            max_tokens: int = 7000
            username: str = payload.socmed_data.username
            logger.info("digital_dna_genai start %s", username)

            tw = payload.socmed_data.tweets
            labels = set(payload.unique_id)
            label_titles = list(payload.title)
            title_to_uid = DNAService._build_title_to_uid_map(payload.title, payload.unique_id)
            uid_to_title = DNAService._build_uid_to_title_map(payload.title, payload.unique_id)
            label_count = len(labels)

            texts = sorted(
                [
                    {
                        "id": i.id,
                        "tweet": i.text,
                        "likes": i.likes,
                        "replies": i.replies,
                        "retweets": i.retweets,
                        "views": i.views or 0,
                        "postedAt": i.postedAt,
                    }
                    for i in tw
                    if not i.isRetweet
                ],
                key=lambda t: t["id"],
            )

            tweet_count = len(texts)
            logger.info("digital_dna_genai user %s stats : tweets count (%s), curr dna count (%s)", username, tweet_count, label_count)
            if(tweet_count < 4):
                raise "INSUFFICIENT_TWEETS"

            response_schema = DNAService._get_base_response_schema()
            client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

            dna_generated_count = 10
            if tweet_count < 10 and tweet_count > 0:
                dna_generated_count = tweet_count

            if tweet_count > 100:
                # collecting sample to count estimated token usage. if exceed, cut texts dump
                # logger.info("digital_dna_genai user %s tweets exceed 100. collecting sample", username)
                sample_size = min(50, len(texts))
                sample_texts = texts[:sample_size]
                sample_json = orjson.dumps(sample_texts)

                sample_resp = client.models.count_tokens(
                    model="gemini-2.5-flash-lite",
                    contents=sample_json
                )

                avg_tokens_per_tweet = sample_resp.total_tokens / sample_size
                estimated_max_tweets = int((max_tokens / avg_tokens_per_tweet) * 0.85)

                texts = texts[:estimated_max_tweets]
                texts_dumps = orjson.dumps(texts)

                token_resp = client.models.count_tokens(
                    model="gemini-2.5-flash-lite",
                    contents=texts_dumps
                )
                token_count = token_resp.total_tokens or 0

                if token_count > max_tokens:
                    adjustment_ratio = max_tokens / token_count
                    texts = texts[:int(len(texts) * adjustment_ratio)]
                    texts_dumps = orjson.dumps(texts)
                    token_count = max_tokens
            else:
                texts_dumps = orjson.dumps(texts)
                token_resp = client.models.count_tokens(
                    model="gemini-2.5-flash-lite",
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

            for tweet_obj in truncated_texts:
                tweet_obj["tweet"] = emoji_to_codepoints(tweet_obj["tweet"])

            texts_dumps = orjson.dumps(truncated_texts)
            tweet_by_id = DNAService._build_tweet_by_id(truncated_texts)
            scorable_tweet_ids = [
                str(tweet["id"])
                for tweet in truncated_texts
                if tweet["tweet"].strip()
            ]
            eligible_tweet_ids = scorable_tweet_ids or [
                str(tweet["id"]) for tweet in truncated_texts
            ]

            label_embeddings = None
            active_schema = response_schema
            enum_titles = label_titles

            if label_count < DNA_TINY_THRESHOLD:
                mode = "tiny"
                label_embeddings = DNAService._build_label_embeddings(label_titles)
                enum_titles = label_titles
                active_schema = DNAService._build_active_schema(
                    response_schema, enum_titles, eligible_tweet_ids
                )
                title_hint = ", ".join(sorted(enum_titles))
                prompt_instruction = (
                    "3. You MUST ONLY use categories from the following allowed list:\n"
                    f"   {title_hint}"
                )
                task_line = (
                    "Classify these tweets using ONLY categories from the allowed list below."
                )
                rule_four = (
                    f"4. Output exactly {dna_generated_count} categories from the allowed list"
                )
                temperature = DNA_TINY_TEMPERATURE
            elif label_count < DNA_CAP_THRESHOLD:
                mode = "discovery"
                enum_titles, label_embeddings = DNAService._build_shortlist(
                    label_titles, truncated_texts, DNA_SHORTLIST_SIZE
                )
                active_schema = DNAService._build_active_schema(
                    response_schema, enum_titles, eligible_tweet_ids
                )
                title_hint = ", ".join(enum_titles)
                prompt_instruction = (
                    "3. You MUST ONLY use categories from the following allowed list:\n"
                    f"   {title_hint}"
                )
                task_line = (
                    "Classify these tweets using ONLY categories from the allowed list below."
                )
                rule_four = (
                    f"4. Output exactly {dna_generated_count} categories from the allowed list"
                )
                temperature = DNA_DISCOVERY_TEMPERATURE
            else:
                mode = "classification"
                enum_titles, label_embeddings = DNAService._build_shortlist(
                    label_titles, truncated_texts, DNA_CLASSIFICATION_SHORTLIST_SIZE
                )
                active_schema = DNAService._build_active_schema(
                    response_schema, enum_titles, eligible_tweet_ids
                )
                title_hint = ", ".join(enum_titles)
                prompt_instruction = (
                    "3. You MUST ONLY use categories from the following list. "
                    "Do NOT create new categories under any circumstances:\n"
                    f"   {title_hint}"
                )
                task_line = (
                    "Classify these tweets into the allowed categories below. "
                    "Do NOT invent new categories."
                )
                rule_four = (
                    f"4. Output up to {dna_generated_count} categories that best describe the tweets"
                )
                temperature = DNA_CLASSIFICATION_TEMPERATURE

            text_prompt = f"""{task_line}

            Tweets:
            {texts_dumps}

            Rules:
            1. Use exact category names from the allowed list only
            2. Avoid semantic redundancy - do not repeat categories
            {prompt_instruction}
            {rule_four}
            5. Percentages must total exactly 100% (approximate is fine; server recalculates)
            6. Each trait needs: category, description (1 paragraph), percentage, tweet_id, and 2 insights
            7. tweet_id must be the exact id field from a tweet in the input; pick a tweet that best represents the category
            8. Insights must directly reference the content of the tweet identified by tweet_id
            9. Do NOT invent new category names; unmatched tweets are analyzed server-side for new DNA proposals

            Don't repeat categories."""

            llm_config = DNAService._build_llm_config(temperature, active_schema)

            text_task = await DNAService._generate_content(
                client, text_prompt, llm_config
            )

            response_text = text_task.text
            response_text = (
                response_text
                .strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            response_text_dict = orjson.loads(response_text)

            dna_dict = DNAService._parse_llm_response(
                response_text_dict, title_to_uid, tweet_by_id
            )
            dna, _ = DNAService._canonicalize_dna(
                dna_dict=dna_dict,
                labels=labels,
                label_titles=label_titles,
                label_embeddings=label_embeddings,
                mode=mode,
                threshold=DNA_SIMILARITY_THRESHOLD,
                title_to_uid=title_to_uid,
                uid_to_title=uid_to_title,
            )

            # 
            sims_data = DNAService._compute_category_tweet_sims(dna, truncated_texts)
            await DNAService._resolve_tweet_samples(
                client, dna, truncated_texts, tweet_by_id, sims_data
            )
            DNAService._apply_tweet_percentages(dna, sims_data)

            new_dna = await DNAService._discover_new_dna_hybrid(
                client=client,
                mode=mode,
                truncated_texts=truncated_texts,
                labels=labels,
                label_titles=label_titles,
                label_embeddings=label_embeddings,
            )

            dna.sort(key=lambda e: e["unique_id"])
            new_dna.sort(key=lambda e: e["unique_id"])

            for entry in dna:
                entry.pop("tweet_id", None)

            return {
                "original_token": token_count,
                "cut_token": current_tokens,
                "free_tweets": len(truncated_texts),
                "dna": dna,
                "new_dna": new_dna,
                "mode": mode,
            }
        except Exception as e:
            logger.exception("digital_dna_genai_err: %s", e)
            raise e

    @staticmethod
    async def generate_dna_image(payload: RequestDigitalDNAImage) -> str:
        try:
            logger.info(f"GENERATE_DNA_IMAGE {payload.title}")

            client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            text_prompt = """
            Convert the input title into a rich, descriptive, safe visual concept for a fantasy-tech badge icon.

            Rules:
            - Output must be a single descriptive phrase or short sentence (8–16 words).
            - Preserve the conceptual meaning through symbolic visual structure.
            - Use only abstract, fantasy-tech, or futuristic visual elements.
            - Do not reference real people, groups, cultures, ideologies, beliefs, or social systems.
            - Do not include emotions, opinions, or narratives.
            - Do not include real-world metaphors (politics, society, religion, culture).
            - Use visual language: shape, structure, energy, light, motion, materials, geometry.
            - Must be safe for image generation systems.

            ====== Example ======
            Input: "Data Privacy"
            Output: "rotating shield lattice around a glowing data crystal"
            ====== End of Example ======
            """

            semantic_transformer = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": text_prompt
                    },
                    {
                        "role": "user",
                        "content": payload.title
                    }
                ],
                seed=42
            )
            visual = semantic_transformer.choices[0].message.content

            logger.info(f"GENERATE_DNA_IMAGE {payload.title} TRANSFORMED >> {visual}")

            image_prompt = f"""You are a visual design AI trained to generate stylized gamified badge icons. 
                Your task is to create a descriptive visual concept for a badge icon based on the input title, using the style described below.

                **Style Guide:**
                - Modern, 3D-styled hexagon or shield-shaped badge
                - Glowing gradients and high contrast colors
                - Smooth shadows and lighting effects
                - Symbolic, fantasy-style icon in the center (e.g. orb, rune, crystal, signal beam, network node, prism)
                - Progress bar or visual indicator optional
                - Similar to mobile RPG/UI badges or NFT gamification

                **Instructions:**
                1. Suggest a neutral color palette.
                2. Describe the icon's shape and what's in the center.
                3. Match the polished, glowing 3D style with fantasy or futuristic elements.
                4. Output a concise description suitable for an image generation AI prompt.
                5. Do not reference real-world groups, movements, systems, or ideologies.

                **Input Prompt:** {visual}
                """

            response = await client.images.generate(
                model="gpt-image-2",
                prompt=image_prompt,
                size="1024x1024",
                quality="low",
                n=1,
            )
            logger.info(f"GENERATE_DNA_IMAGE {payload.title} IMAGE GENERATED - REMOVING BACKGROUND")

            image_base64 = response.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)
            input_image = Image.open(BytesIO(image_bytes))
            nobg_image = remove(input_image)

            average_hex = get_average_hex_color(nobg_image)

            buffer = BytesIO()
            nobg_image.save(buffer, format="PNG")
            buffer.seek(0)
            image_b64 = base64.b64encode(buffer.read()).decode("utf-8")

            logger.info(f"GENERATE_DNA_IMAGE {payload.title} IMAGE FINISH")
            image = {
                "image_b64": image_b64,
                "background_hex": average_hex
            }

            return image
        except Exception as e:
            logger.exception("generate_dna_image_err: %s", e)
            raise
