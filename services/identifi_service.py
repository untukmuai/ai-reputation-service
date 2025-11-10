


import math
from statistics import mean
from models.requests.identifi_request import RequestIdentifiScore
from models.responses.base_response import BaseResponse, ErrorResponse
from services.somnia_referral_service import SomniaReferralService


class IdentifiScore:
    """Service to handle identifi score calculation"""

    @staticmethod
    async def calculate_identifi_log(payload: RequestIdentifiScore):
        try:
            followers_count = payload.public_metrics.followers_count
            avg_views    = round(mean(i.views if i.views is not None else 0 for i in payload.tweets))
            avg_likes    = round(mean(i.likes if i.likes is not None else 0 for i in payload.tweets))
            avg_replies  = round(mean(i.replies if i.replies is not None else 0 for i in payload.tweets))
            avg_retweets = round(mean(i.retweets if i.retweets is not None else 0 for i in payload.tweets))
            badges_minted = payload.badges_minted
            quest_completed = payload.quest_completed
            total_badges_reward = payload.total_badges_reward
            referral_count = 0

            if payload.address:
                referral_service = SomniaReferralService()
                referral_count = referral_service.get_referral_count(payload.address)

            total_social_score = followers_count + avg_views
            total_reputation_score = avg_likes + avg_replies + avg_retweets
            total_onchain_score = total_badges_reward + referral_count
            log_social_score = math.log10(total_social_score + 1) * 50
            log_reputation_score = math.log10(total_reputation_score + 1) * 75
            log_onchain_score = math.log10(total_onchain_score + 1) * 50
            log_governance_score = math.log10(0 + 1) * 25

            return {
                "social_score": {
                    "followers_count": followers_count,
                    "avg_views": avg_views,
                    "total_social_score": total_social_score,
                    "log_social_score": log_social_score,
                },
                "reputation_score": {
                    "avg_likes": avg_likes,
                    "avg_replies": avg_replies,
                    "avg_retweets": avg_retweets,
                    "total_reputation_score": total_reputation_score,
                    "log_reputation_score": log_reputation_score,
                },
                "governance_score": {
                    "total_governance_score": 0,
                    "log_governance_score": log_governance_score,
                },
                "onchain_score": {
                    "badges_minted": badges_minted,
                    "quest_completed": quest_completed,
                    "total_badges_reward": total_badges_reward,
                    "referral_count": referral_count,
                    "total_onchain_score": total_onchain_score,
                    "log_onchain_score": log_onchain_score,
                },
                "identifi_score": total_social_score + total_reputation_score + total_onchain_score + 1,
                "log_identifi_score": log_social_score + log_reputation_score + log_onchain_score + log_governance_score
            }
        except Exception as e:
            raise e