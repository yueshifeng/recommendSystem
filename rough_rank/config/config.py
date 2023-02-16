
from .feature_id import FEATURE_ID, ALL_FEATURE_ID

# feature_name->slot_id
FEATURE_SLOT = {
    # user
    'video_finish_videoid': 2,
    'video_finish_hashtags': 4,
    'video_finish_author_id': 5,
    'video_finish_music_id': 6,
    'videorec_userprofile_Age': 1567,
    'videorec_userprofile_user_id': 1568,
    'videorec_userprofile_Gender': 1570,
    'videorec_userprofile_Birthday_Month': 1571,
    'videorec_userprofile_Language': 1572,
    'videorec_userprofile_Address_State': 1574,
    'videorec_userprofile_Address_City': 1575,
    'videorec_userprofile_Phone_Brand': 1576,
    'videorec_userprofile_Phone_Model': 1577,
    'videorec_userprofile_Phone_OS': 1578,
    'videorec_userprofile_Shopee_Rewards_Tier': 1579,
    'videorec_userprofile_Consumption_Level_in_Last_30_Days': 1582,
    'videorec_userprofile_EcomMostPurchasedCate1List': 1586,
    'videorec_userprofile_AppNameList': 1589,
    'videorec_userprofile_Video_Phone_Model_Price_Level': 1736,
    'videorec_userprofile_Age_bucket': 2039,
    'video_sequence_finish_hashTag': 2123,
    'video_sequence_finish_videoid': 2125,
    'video_sequence_finish_musicid': 2127,
    'video_sequence_finish_authorid': 2128,
    'video_sequence_finish_contentL1': 2130,
    'video_sequence_finish_contentL2': 2131,
    'video_sequence_share_shopee_video_musicid': 2148,
    'video_sequence_share_shopee_video_authorid': 2150,
    'video_sequence_like_video_musicid': 2151,
    'video_sequence_share_shopee_video_videoid': 2153,
    'video_sequence_like_video_contentL1': 2154,
    'video_sequence_share_shopee_video_contentL2': 2155,
    "video_context_bundle": 2597,

    # item
    'video_videoid_v2': 1591,
    'video_country_v2': 1592,
    'video_authorid_v2': 1593,
    'video_source': 1594,
    'video_state': 1595,
    'video_language_v2': 1601,
    'video_musicid_v2': 1614,
    'video_music_authorname': 1616,
    'video_crawler_authorid': 1624,
    'video_content_l1_cate_id': 1737,
    'video_content_l2_cate_id': 1738,
    'video_music_duration_bucket_v1': 2040,
    'video_width_bucket': 2041,
    'video_heigth_bucket': 2042,
    'video_videosize_bucket': 2043,
    'video_duration_bucket': 2044,
    'video_crawler_likecount_bucket': 2045,
    'video_crawler_commentcount_bucket': 2046,
    'video_crawler_impressioncount_bucket': 2049,
}

ALL_FEATURE_SLOT = {slot_id for slot_id in FEATURE_SLOT.values()}

def get_feature_id(feature_name):
    if feature_name in FEATURE_ID:
        return str(FEATURE_ID[feature_name])
    if feature_name in FEATURE_SLOT:
        return str(FEATURE_SLOT[feature_name])
    raise ValueError("feature: {} not found".format(feature_name))

ALL_FEATURE_ID_2_SLOT = {get_feature_id(feature_name) : slot_id for feature_name, slot_id in FEATURE_SLOT.items()}

USER_FEATURES = ['video_context_bundle', 'video_finish_videoid', 'video_finish_hashtags', 'video_finish_author_id', 'video_finish_music_id', 'videorec_userprofile_Age', 'videorec_userprofile_user_id', 'videorec_userprofile_Gender', 'videorec_userprofile_Birthday_Month', 'videorec_userprofile_Language', 'videorec_userprofile_Address_State', 'videorec_userprofile_Address_City', 'videorec_userprofile_Phone_Brand', 'videorec_userprofile_Phone_Model', 'videorec_userprofile_Phone_OS', 'videorec_userprofile_Shopee_Rewards_Tier', 'videorec_userprofile_Consumption_Level_in_Last_30_Days', 'videorec_userprofile_EcomMostPurchasedCate1List', 'videorec_userprofile_AppNameList', 'videorec_userprofile_Video_Phone_Model_Price_Level', 'videorec_userprofile_Age_bucket', 'video_sequence_finish_hashTag', 'video_sequence_finish_videoid', 'video_sequence_finish_musicid', 'video_sequence_finish_authorid', 'video_sequence_finish_contentL1', 'video_sequence_finish_contentL2', 'video_sequence_share_shopee_video_musicid', 'video_sequence_share_shopee_video_authorid', 'video_sequence_like_video_musicid', 'video_sequence_share_shopee_video_videoid', 'video_sequence_like_video_contentL1', 'video_sequence_share_shopee_video_contentL2']
USER_FEATURE_IDS = [get_feature_id(feature_name) for feature_name in USER_FEATURES]

ITEM_FEATURES = ['video_videoid_v2', 'video_country_v2', 'video_authorid_v2', 'video_source', 'video_state', 'video_language_v2', 'video_musicid_v2', 'video_music_authorname', 'video_crawler_authorid', 'video_content_l1_cate_id', 'video_content_l2_cate_id', 'video_music_duration_bucket_v1', 'video_width_bucket', 'video_heigth_bucket', 'video_videosize_bucket', 'video_duration_bucket', 'video_crawler_likecount_bucket', 'video_crawler_commentcount_bucket', 'video_crawler_impressioncount_bucket']
ITEM_FEATURE_IDS = [get_feature_id(feature_name) for feature_name in ITEM_FEATURES]

USER_OUTPUT_DIM = 16
ITEM_OUTPUT_DIM = 16

# task_names = ['staytime', 'shortplay', 'longplay']

shuffle_buffer_size = 16
