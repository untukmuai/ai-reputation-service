


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
            avg_views    = round(mean(i.views if i.views is not None else 0 for i in payload.tweets))
            avg_likes    = round(mean(i.likes if i.likes is not None else 0 for i in payload.tweets))
            avg_replies  = round(mean(i.replies if i.replies is not None else 0 for i in payload.tweets))
            avg_retweets = round(mean(i.retweets if i.retweets is not None else 0 for i in payload.tweets))
            badges_minted = payload.badges_minted
            quest_completed = payload.quest_completed
            referral_count = 0

            if payload.address:
                referral_service = SomniaReferralService()
                referral_count = await referral_service.get_referral_count_async(payload.address)


            #network/social score calc
            follower_score = (math.log10(payload.public_metrics.followers_count + 1)) * 50
            view_score =  (math.log10(avg_views + 1)) * 50
            social_score = follower_score + view_score

            #engagement score calc
            total_engagement = avg_likes + avg_replies + avg_retweets
            engagement_score = math.log10(total_engagement + 1) * 75

            #on chain score calc
            referral_score = math.log10(referral_count + 1) * 30
            badges_score = payload.total_badges_reward * 0.4
            onchain_score = badges_score + referral_score

            #identifi final cacl
            final_identifi_score = social_score + engagement_score + onchain_score


            
            # total_social_score = followers_count + avg_views
            # total_reputation_score = avg_likes + avg_replies + avg_retweets
            # log_social_score = math.log10(total_social_score + 1) * 50
            # log_reputation_score = math.log10(total_reputation_score + 1) * 75
            # log_onchain_score = math.log10(total_onchain_score + 1) * 50
            # log_governance_score = math.log10(0 + 1) * 25

            return {
                "social_score": {
                    "followers_count": payload.public_metrics.followers_count,
                    "avg_views": avg_views,
                    "final": social_score,
                    # "log_social_score": log_social_score,
                },
                "reputation_score": {
                    "avg_likes": avg_likes,
                    "avg_replies": avg_replies,
                    "avg_retweets": avg_retweets,
                    "final": engagement_score,
                    # "log_reputation_score": log_reputation_score,
                },
                "governance_score": {
                    "final": 0,
                    # "log_governance_score": log_governance_score,
                },
                "onchain_score": {
                    "badges_minted": badges_minted,
                    "quest_completed": quest_completed,
                    "total_badges_reward": payload.total_badges_reward,
                    "referral_count": referral_count,
                    "final": onchain_score,
                    # "log_onchain_score": total_onchain_score,
                },
                "identifi_score": final_identifi_score
                # "log_identifi_score": log_social_score + log_reputation_score + total_onchain_score + log_governance_score
            }
        except Exception as e:
            print(str(e))
            raise e