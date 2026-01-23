import logging
import math
from statistics import mean
from models.requests.identifi_request import RequestIdentifiScore, RequestIdentifiScoreV2
from models.requests.tweet_request import Tweets
from services.identifi_util_service import IdentifiScoreUtil
from services.somnia_referral_service import SomniaReferralService
import math
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

logger = logging.getLogger(__name__)

tfidf = TfidfVectorizer(min_df=1,
    max_features=2000,
    ngram_range=(1, 2) )
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class IdentifiScore:
    """Service to handle identifi score calculation"""

    # Indonesian stopwords (extend as needed)
    ID_STOPWORDS = set("""
    yang untuk pada dengan tidak dari bahwa karena oleh sebagai dalam
    itu ada kami mereka dia ini kamu saja bisa atau jadi kalau maka
    """.split())

    # English stopwords
    EN_STOPWORDS = set("""
    the a an is are was were be to of and or in on for as by with this that it
    """.split())

    MEDIA_RICHNESS_WEIGHT = {
        "image": float(os.getenv('MEDIA_RICHNESS_WEIGHT_IMAGE', '0.3')),
        "video": float(os.getenv('MEDIA_RICHNESS_WEIGHT_VIDEO', '0.6'))
    }

    SIMILARITY_SPAM_THRESHOLDS = float(os.getenv('SIMILARITY_SPAM_THRESHOLDS', '0.9'))
    SPAM_PENALTY_MULTIPLIER = int(os.getenv('SPAM_PENALTY_MULTIPLIER', '7'))

    @staticmethod
    def get_media_richness_score(photos: int, videos: int):
        cap=0.6
        k=1.2
        diversity_video_image_bonus=0.05
        max_allowed=0.65

        sum = IdentifiScore.MEDIA_RICHNESS_WEIGHT.get("image") * photos + IdentifiScore.MEDIA_RICHNESS_WEIGHT.get("video") * videos
        raw = cap * (1 - math.exp(-k * (sum / cap)))
        diversity = diversity_video_image_bonus if (photos > 0 and videos > 0) else 0.0
        score = min(raw + diversity, max_allowed)
        return round(score, 2) * 10
    
    @staticmethod
    def get_readability_score(text: str) -> float:
        if not text or not text.strip():
            return 0.3  # minimum score

        lang = IdentifiScoreUtil.detect_language_safe(text)
        tokens = IdentifiScoreUtil.tokenize(text)
        token_count = len(tokens)

        words = [t for t in tokens if re.match(r"\w+", t)]
        word_count = len(words)

        if word_count == 0:
            return 0.3

        # ---------- BASE METRICS ----------
        sentences = IdentifiScoreUtil.count_sentences(text)
        avg_sentence_len = word_count / sentences
        avg_word_len = sum(len(w) for w in words) / word_count
        max_word_len = max(len(w) for w in words)

        # Stopwords
        stopwords = IdentifiScore.ID_STOPWORDS if lang.startswith("id") else IdentifiScore.EN_STOPWORDS
        stopword_ratio = sum(1 for w in words if w.lower() in stopwords) / word_count

        # ---------- NOISE METRICS ----------
        hashtag_density = len([t for t in tokens if t.startswith("#")]) / token_count
        emoji_density = IdentifiScoreUtil.emoji_count(text) / token_count
        punct_complexity = len(re.findall(r"[,;:—]", text)) / token_count

        # ---------- NATURALNESS METRICS ----------
        # Token diversity: low diversity = unnatural (spammy)
        unique_tokens = len(set(words))
        token_diversity = unique_tokens / word_count
        low_diversity_penalty = 1 - token_diversity

        # Very long word penalty
        long_word_penalty = min(max_word_len / 20, 1.0)

        # Vowel penalty (new)
        vowel_penalties = sum(1 - min(IdentifiScoreUtil.vowel_ratio(w) * 2, 1.0) for w in words)
        vowel_penalty = vowel_penalties / word_count

        # Zero-stopword penalty (for non-short text)
        zero_stopword_penalty = 1.0 if (stopword_ratio == 0 and len(text) > 20) else 0.0

        # ---------- NORMALIZATION HELPERS ----------
        def inverse_norm(x, cap=30):
            x = min(x, cap)
            return 1 / (1 + x)

        f_sentence = inverse_norm(avg_sentence_len)
        f_word = inverse_norm(avg_word_len, cap=15)

        # ---------- FINAL RAW SCORE ----------
        raw = (
            # Structure (45%)
            0.25 * f_sentence +
            0.20 * f_word +

            # Naturalness (32%)
            -0.15 * long_word_penalty +
            -0.15 * vowel_penalty +
            -0.07 * low_diversity_penalty +

            # Noise (12%)
            -0.10 * hashtag_density +
            -0.03 * emoji_density +
            -0.05 * punct_complexity +

            # Stopwords (11%)
            0.15 * stopword_ratio -
            0.08 * zero_stopword_penalty
        )

        # Normalize raw score into 0–1
        raw_norm = max(0.0, min(1.0, raw + 0.5))

        # Map to readability band 0.3 → 0.9
        final = 0.3 + 0.6 * raw_norm
        return round(final, 4)

    @staticmethod
    def get_originality_score(tweet: Tweets) -> float:
        if tweet.isRetweet:
            return 1.0
        else:
            return 0.0

    @staticmethod
    def get_link_spam_score(text: str):
        urls = IdentifiScoreUtil.extract_urls(text)

        if not urls:
            return 0.0  # no link, no penalty

        max_penalty = 0.0
        for url in urls:
            spammy, severity = IdentifiScoreUtil.is_spam_url(url)
            if spammy:
                max_penalty = max(max_penalty, severity)

        return -max_penalty  # negative score for spam

    @staticmethod
    def prepare_spam_features(tweets_list, embedder, tfidf):
        tfidf_matrix = tfidf.fit_transform(tweets_list)
        embed_matrix = embedder.encode(tweets_list, normalize_embeddings=True, show_progress_bar=False)
        return tfidf_matrix, embed_matrix
    
    @staticmethod
    def get_spam_score(current_tweet, index, tfidf_matrix, embed_matrix, embedder, tfidf):
        current_tfidf = tfidf.transform([current_tweet])
        sims_tfidf = cosine_similarity(current_tfidf, tfidf_matrix)[0]

        # exclude curr tweet tfidf
        sims_tfidf[index] = -1
        sim_tfidf = sims_tfidf.max()

        # embedding similarity
        current_embed = embedder.encode([current_tweet], normalize_embeddings=True, show_progress_bar=False)
        sims_embed = util.cos_sim(current_embed, embed_matrix).cpu().numpy()[0]

        # exclude curr tweet similarity
        sims_embed[index] = -1
        sim_embed = sims_embed.max()

        # penalties
        freq_penalty = float((sim_tfidf + sim_embed) / 2 > 0.75)

        text_len = len(current_tweet)
        short_penalty = 1 if text_len <= 4 else 0.5 if text_len <= 8 else 0

        z = (
            1.2 * sim_embed +
            0.8 * sim_tfidf +
            0.3 * freq_penalty +
            0.4 * short_penalty
        )

        return float(1 / (1 + math.exp(-z)))

    @staticmethod
    async def calculate_identifi_v2(payload: RequestIdentifiScoreV2):
        logger.info(f'IDENTIFI_SCORE_V2 {payload.username} START')
        try:
            t0 = time.time()

            tweet_obj_list = []
            seen_tweet_ids = set()
            tweets_list = []
            tweet_set_detect_spam = set()
            tweet_text_index_map = {}  # maps original index to filtered index

            # ensure unique tweet id
            for tweet in payload.tweets:
                if tweet.id not in seen_tweet_ids:
                    current_index = len(tweet_obj_list)  # store before appending
                    tweet_obj_list.append(tweet)
                    seen_tweet_ids.add(tweet.id)
                    tweet.text = IdentifiScoreUtil.clean_tweet(tweet.text)
                    if len(tweet.text) > 0:
                        tweet_text_index_map[current_index] = len(tweets_list)  # map original to filtered
                        tweets_list.append(tweet.text)
                        tweet_set_detect_spam.add(tweet.id)
            tweet_len = len(tweet_obj_list)

            # accumulators
            readability_score = 0
            spammy_link_score = 0
            image_count = 0
            video_count = 0
            views_count = 0
            likes_count = 0
            retweets_count = 0
            replies_count = 0
            spam_sim_arr = []
            spam_sim_tweets_arr = []
            original_count = 0
            detect_text = ""
            
            # Prepare spam detection features from all tweets
            if len(tweets_list) > 0:
                tfidf_matrix, embed_matrix = IdentifiScore.prepare_spam_features(tweets_list, embedder, tfidf)

            # calculate scores
            for index, tweet in enumerate(tweet_obj_list):
                # content
                readability_score += IdentifiScore.get_readability_score(tweet.text)
                image_count += len(tweet.photos)
                video_count += len(tweet.videos)

                # spam link, originality, text detect collection
                if tweet.isRetweet == False:
                    # engagement
                    views_count += tweet.views if tweet.views is not None else 0
                    likes_count += tweet.likes if tweet.likes is not None else 0
                    retweets_count += tweet.retweets if tweet.retweets is not None else 0
                    replies_count += tweet.replies if tweet.replies is not None else 0

                    # spam similarity score. only detect spam if tweet is marked 
                    if tweet.id in tweet_set_detect_spam:
                        filtered_index = tweet_text_index_map[index]
                        sim_score = IdentifiScore.get_spam_score(tweet.text, filtered_index, tfidf_matrix, embed_matrix, embedder, tfidf)
                        # print(f"{tweet.text} || SIM SCORE: {sim_score}")
                        if sim_score >= IdentifiScore.SIMILARITY_SPAM_THRESHOLDS:
                            spam_sim_arr.append(sim_score)
                            spam_sim_tweets_arr.append({
                                "tweet": tweet.text,
                                "id": tweet.id
                            })

                    spammy_link_score += IdentifiScore.get_link_spam_score(tweet.text)
                    original_count += 1
                    detect_text += tweet.text + "\n\n"

                # elapsed_ms = (time.time() - start_time) * 1000

            avg_views = 0
            avg_likes = 0
            avg_retweets = 0
            avg_replies = 0

            if original_count > 0:
                avg_views = round(views_count / original_count, 2)
                avg_likes = round(likes_count / original_count, 2)
                avg_retweets = round(retweets_count / original_count, 2)
                avg_replies = round(replies_count / original_count, 2)
            else:
                logger.info(f'IDENTIFI_SCORE_V2 {payload.username} has 0 original tweet. average engagement metric set to 0')

            # network score
            follower_score = round(math.log(payload.public_metrics.followers_count + 1) * 50, 2)
            view_score = round(math.log(avg_views + 1) * 50, 2)
            network_score = round(follower_score + view_score, 2)

            # engagement score
            sum_engagement = round(avg_likes + avg_retweets + avg_replies, 2)
            engagement_score = round(math.log(sum_engagement + 1) * 75, 2)

            #feedback score
            feedback_score = 0

            if payload.voters:
                feedback_weight_sum = 0
                feedback_value = 0
                for voter in payload.voters:
                    weight = math.log(voter.followers + 10) * math.sqrt(voter.twitter_account_age_days / 30) * voter.quality_score
                    feedback_weight_sum += weight
                    value = 1 if voter.vote == "up" else -1
                    feedback_value += value * weight 
                    
                if feedback_weight_sum > 0:
                    feedback_score = feedback_value / feedback_weight_sum
                else:
                    feedback_score = 0

            # quality score
            originality_score = round((original_count / tweet_len) * 100, 2)
            media_richness_score = IdentifiScore.get_media_richness_score(image_count, video_count)
            readability_score = round(readability_score, 2)
            content_score = round(readability_score + media_richness_score, 2)

            if(len(spam_sim_arr) > 0 and original_count > 0):
                spam_ratio = len(spam_sim_arr) / original_count
                avg_sim = sum(spam_sim_arr) / len(spam_sim_arr)
                base_penalty = -(max(0, avg_sim - IdentifiScore.SIMILARITY_SPAM_THRESHOLDS) / (1 - IdentifiScore.SIMILARITY_SPAM_THRESHOLDS)) * 100
                count_multiplier = 1 + (spam_ratio * IdentifiScore.SPAM_PENALTY_MULTIPLIER) 
                excess_sim = max(0, avg_sim - IdentifiScore.SIMILARITY_SPAM_THRESHOLDS)
                exponential_factor = (excess_sim / (1 - IdentifiScore.SIMILARITY_SPAM_THRESHOLDS)) ** 2
                spam_penalty = base_penalty * count_multiplier * (1 + exponential_factor)
            else:
                spam_penalty = 0
                # print(f"{spam_penalty}")

            # spam_penalty = max(spam_penalty, -(tweet_len / 2)) enable if want to be capped

            spammy_link_score *= 100
            spam_penalty = round(spam_penalty, 2)
            spammy_link_score = round(spammy_link_score, 2)
            behaviour_score = originality_score + spam_penalty + spammy_link_score
            quality_score = round(content_score + behaviour_score, 2)

            # on chain
            referral_count = 0
            referral_score = 0
            badges_score = 0
            on_chain_score = 0
            
            # if payload.address:
            #     referral_service = SomniaReferralService()
            #     referral_count = await referral_service.get_referral_count_async(payload.address)

            if(referral_count > 0):
                referral_score = math.log10(referral_count + 1) * 30
                
            badges_score = payload.total_badges_reward * 0.4
            on_chain_score = round(referral_score + badges_score, 2)

            # final
            identifi_score = round(network_score + engagement_score + quality_score + on_chain_score, 2)

            finished_ms = (time.time() - t0) * 1000
            logger.info(f"IDENTIFI_SCORE_V2 {payload.username} FINISHED in {finished_ms:.2f} ms\n")

            return {
                "network": {
                    "followers_count": payload.public_metrics.followers_count,
                    "follower_score": follower_score,
                    "average_views": avg_views,
                    "view_score": view_score,
                    "overall": network_score
                },
                "engagement": {
                    "average_likes": avg_likes,
                    "average_retweets": avg_retweets,
                    "average_replies": avg_replies,
                    "sum_engagement": sum_engagement,
                    "overall": engagement_score
                },
                "feedback": {
                    "overall": feedback_score,
                },
                "quality": {
                    "content_score": {
                        "media_richness": media_richness_score,
                        "readability": readability_score,
                        "overall": content_score
                    },
                    "behaviour_score": {
                        "spam_tweet_penalty": spam_penalty,
                        "spam_keyword_penalty": spammy_link_score,
                        "originality": originality_score,
                        "overall": behaviour_score
                    },
                    "overall": quality_score
                },
                "onchain":{
                    "referral": referral_count,
                    "badges_minted_count": payload.badges_minted,
                    "badges_reward_accumulated": payload.total_badges_reward,
                    "badges_score": badges_score,
                    "overall": on_chain_score
                },
                "identifi": identifi_score,
                "proof": {
                    "spam_tweets": spam_sim_tweets_arr
                }
            }
        except Exception as e:
            logger.exception(f"IDENTIFI_SCORE_V2_ERR {payload.username} {e}")
            raise
    
    # deprecated function
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
            logger.exception("calculate_identifi_log_err: %s", e)
            raise 
