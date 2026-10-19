#!/usr/bin/env python3
"""Common augmentation policy catalog derived once from measured platform CSVs.

This file is the single source of truth for both offline and online platform-aware
augmentation. Runtime code does NOT read measurement.csv again. Instead it samples
from pre-built augmentation recipes materialized here.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Optional

PLATFORMS = ('facebook', 'instagram', 'kakao')
MODALITIES = ('image', 'video')

PLATFORM_MIXTURE_DEFAULT = {
    'clean': 0.25,
    'facebook': 0.25,
    'instagram': 0.25,
    'kakao': 0.25,
}

QUALITY_WEIGHTS_DEFAULT = {
    ('facebook', 'image'): {'high': 1/3, 'mid': 1/3, 'low': 1/3},
    ('facebook', 'video'): {'high': 0.50, 'mid': 0.25, 'low': 0.25},
    ('instagram', 'image'): {'shared': 1.0},
    ('instagram', 'video'): {'shared': 1.0},
    ('kakao', 'image'): {'original': 1/3, 'normal': 1/3, 'low': 1/3},
    ('kakao', 'video'): {'single': 1.0},
}

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_SUFFIXES = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm'}

EMBEDDED_SOURCE_SHA256 = {'facebook_high_image_measurement.csv': '5bb151ab4602933434d02ec461efa2278d09459708d6d9e5e37fcb245136dde9', 'facebook_high_video_measurement.csv': 'b75e622bbe73aa3126eabd594c3b76856f5baa3cfc319abe0a22a1ee00452a8e', 'facebook_low_image_measurement.csv': '0fddd9ea8b147c1e6dbb5242f20f9655132edbc536f3533300a81f9b05cabb55', 'facebook_low_video_measurement.csv': 'f5b6803b38704f06057f237b3a93ecc4ca1f81135308ae9498ccc768b30ab239', 'facebook_mid_image_measurement.csv': '43c83c30e8cf42fdc102a7536eb7dd355a22496a966b756bea727aed19eba283', 'facebook_mid_video_measurement.csv': '304d8acac96b9c3cdcc78df55b0c598ea9d58cd01f8602c04185e2c1ccdbdd03', 'instagram_high_image_measurement.csv': 'aa0a35bcd7b883e47556069e4370fc564e65c3dccbcb19013f4a048c9d1bda93', 'instagram_high_video_measurement.csv': '101624e7ee01201cbde78bec270359df7df9420489a442cd32d08c81ca880d7b', 'kakao_low_image_measurement.csv': '8a99a9ae954821174d36a557fce4476f0b3baa1442e780cf67a3ac5e2654f319', 'kakao_normal_image_measurement.csv': '6e3e595a2f6acb15f38e60c3d9cc194154089f16e09b0307fd48b40b347ebf53', 'kakao_original_image_measurement.csv': '08b6287f25cb4d1b8bdad31721ab92f71fc697831099a0e786baf345e389d2a2', 'kakao_video_measurement.csv': '953a36be0a7e4f7d16bd7e8f29005c2c0e3be03bd509f304ec938138f8e2a386'}
# Pre-built joint transformation recipes, generated once from measurement CSVs.
# Each tuple is one FIXED augmentation combination with all correlated fields.
# The runtime samples a catalog recipe; it does not read/measure any CSV.
RECIPE_FIELDS = ('id', 'platform', 'modality', 'quality', 'band', 'orientation', 'weight', 'source_row_ordinal', 'reencode_verdict', 'orig_width', 'orig_height', 'down_width', 'down_height', 'crop_applied', 'down_estimated_jpeg_qf', 'down_chroma_subsampling', 'down_jpeg_progressive', 'brightness_offset_b', 'contrast_gain_a', 'effective_crf_interp', 'bitrate_ratio', 'down_codec_name', 'down_profile', 'down_avg_frame_rate', 'down_gop_mean', 'down_has_b_frames', 'temporal_offset_frames')
RECIPE_TUPLES = (
    ('FACEBOOK_IMAGE_HIGH_LE1080_LAND_01', 'facebook', 'image', 'high', 'le1080', 'land', 1, 4, 'container_or_metadata_only', 1080, 608, 1080, 608, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE1080_LAND_02', 'facebook', 'image', 'high', 'le1080', 'land', 1, 5, 'container_or_metadata_only', 1080, 608, 1080, 608, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE1080_LAND_03', 'facebook', 'image', 'high', 'le1080', 'land', 1, 6, 'container_or_metadata_only', 1080, 608, 1080, 608, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE1080_LAND_04', 'facebook', 'image', 'high', 'le1080', 'land', 1, 7, 'container_or_metadata_only', 1080, 608, 1080, 608, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE1080_LAND_05', 'facebook', 'image', 'high', 'le1080', 'land', 1, 8, 'container_or_metadata_only', 1080, 608, 1080, 608, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE1080_LAND_06', 'facebook', 'image', 'high', 'le1080', 'land', 1, 9, 'container_or_metadata_only', 1080, 594, 1080, 594, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE1080_LAND_07', 'facebook', 'image', 'high', 'le1080', 'land', 1, 11, 'container_or_metadata_only', 1080, 719, 1080, 719, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE1080_PORT_01', 'facebook', 'image', 'high', 'le1080', 'port', 1, 1, 'container_or_metadata_only', 844, 1080, 844, 1080, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE1080_PORT_02', 'facebook', 'image', 'high', 'le1080', 'port', 1, 2, 'container_or_metadata_only', 720, 1080, 720, 1080, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE1080_PORT_03', 'facebook', 'image', 'high', 'le1080', 'port', 1, 3, 'container_or_metadata_only', 767, 1080, 767, 1080, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE2048_LAND_01', 'facebook', 'image', 'high', 'le2048', 'land', 1, 10, 'resized_and_recompressed', 2048, 1365, 1440, 960, False, 93.0, '4:2:0', True, -0.000744, 0.999829, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE2048_LAND_02', 'facebook', 'image', 'high', 'le2048', 'land', 1, 14, 'resized_and_recompressed', 2048, 1365, 1440, 960, False, 93.0, '4:2:0', True, -0.000934, 1.000404, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE2048_LAND_03', 'facebook', 'image', 'high', 'le2048', 'land', 1, 15, 'resized_and_recompressed', 2048, 1366, 1439, 960, False, 93.0, '4:2:0', True, -0.001073, 1.00058, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE2048_LAND_04', 'facebook', 'image', 'high', 'le2048', 'land', 1, 18, 'resized_and_recompressed', 2048, 1365, 1440, 960, False, 93.0, '4:2:0', True, -0.000855, 1.00009, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE2048_PORT_01', 'facebook', 'image', 'high', 'le2048', 'port', 1, 12, 'container_or_metadata_only', 1365, 2048, 1365, 2048, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE2048_PORT_02', 'facebook', 'image', 'high', 'le2048', 'port', 1, 13, 'container_or_metadata_only', 1536, 2048, 1536, 2048, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE2048_PORT_03', 'facebook', 'image', 'high', 'le2048', 'port', 1, 16, 'container_or_metadata_only', 1152, 2048, 1152, 2048, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE2048_PORT_04', 'facebook', 'image', 'high', 'le2048', 'port', 1, 17, 'container_or_metadata_only', 1357, 2048, 1357, 2048, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE2048_PORT_05', 'facebook', 'image', 'high', 'le2048', 'port', 1, 19, 'container_or_metadata_only', 1365, 2048, 1365, 2048, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE2048_PORT_06', 'facebook', 'image', 'high', 'le2048', 'port', 1, 20, 'container_or_metadata_only', 1633, 2048, 1633, 2048, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE512_PORT_01', 'facebook', 'image', 'high', 'le512', 'port', 1, 21, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE512_PORT_02', 'facebook', 'image', 'high', 'le512', 'port', 1, 22, 'container_or_metadata_only', 444, 512, 444, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE512_PORT_03', 'facebook', 'image', 'high', 'le512', 'port', 1, 23, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE512_PORT_04', 'facebook', 'image', 'high', 'le512', 'port', 1, 24, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE512_PORT_05', 'facebook', 'image', 'high', 'le512', 'port', 1, 25, 'container_or_metadata_only', 444, 512, 444, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE512_PORT_06', 'facebook', 'image', 'high', 'le512', 'port', 1, 26, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE512_PORT_07', 'facebook', 'image', 'high', 'le512', 'port', 1, 27, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE512_PORT_08', 'facebook', 'image', 'high', 'le512', 'port', 1, 28, 'container_or_metadata_only', 444, 512, 444, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE512_PORT_09', 'facebook', 'image', 'high', 'le512', 'port', 1, 29, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_HIGH_LE512_PORT_10', 'facebook', 'image', 'high', 'le512', 'port', 1, 30, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE1080_LAND_01', 'facebook', 'image', 'low', 'le1080', 'land', 1, 4, 'resized_and_recompressed', 1080, 608, 640, 360, False, 93.0, '4:2:0', True, -0.000599, 0.999668, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE1080_LAND_02', 'facebook', 'image', 'low', 'le1080', 'land', 1, 5, 'resized_and_recompressed', 1080, 608, 640, 360, False, 93.0, '4:2:0', True, -0.00132, 0.999886, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE1080_LAND_03', 'facebook', 'image', 'low', 'le1080', 'land', 1, 6, 'resized_and_recompressed', 1080, 608, 640, 360, False, 93.0, '4:2:0', True, -0.001843, 1.00146, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE1080_LAND_04', 'facebook', 'image', 'low', 'le1080', 'land', 1, 7, 'resized_and_recompressed', 1080, 608, 640, 360, False, 93.0, '4:2:0', True, -0.001609, 1.000328, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE1080_LAND_05', 'facebook', 'image', 'low', 'le1080', 'land', 1, 8, 'resized_and_recompressed', 1080, 608, 640, 360, False, 93.0, '4:2:0', True, -0.00164, 1.000362, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE1080_LAND_06', 'facebook', 'image', 'low', 'le1080', 'land', 1, 9, 'resized_and_recompressed', 1080, 594, 716, 394, False, 93.0, '4:2:0', True, -0.001254, 1.000401, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE1080_LAND_07', 'facebook', 'image', 'low', 'le1080', 'land', 1, 11, 'resized_and_recompressed', 1080, 719, 552, 367, False, 93.0, '4:2:0', True, -0.001578, 0.999708, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE1080_PORT_01', 'facebook', 'image', 'low', 'le1080', 'port', 1, 1, 'resized_and_recompressed', 844, 1080, 480, 614, False, 93.0, '4:2:0', True, -0.001523, 1.001238, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE1080_PORT_02', 'facebook', 'image', 'low', 'le1080', 'port', 1, 2, 'resized_and_recompressed', 720, 1080, 480, 720, False, 93.0, '4:2:0', True, -0.002546, 1.000126, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE1080_PORT_03', 'facebook', 'image', 'low', 'le1080', 'port', 1, 3, 'resized_and_recompressed', 767, 1080, 480, 676, False, 93.0, '4:2:0', True, -0.001124, 0.999885, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE2048_LAND_01', 'facebook', 'image', 'low', 'le2048', 'land', 1, 10, 'resized_and_recompressed', 2048, 1365, 552, 368, False, 93.0, '4:2:0', True, -0.000897, 1.000142, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE2048_LAND_02', 'facebook', 'image', 'low', 'le2048', 'land', 1, 14, 'resized_and_recompressed', 2048, 1365, 552, 368, False, 93.0, '4:2:0', True, -0.001115, 1.000784, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE2048_LAND_03', 'facebook', 'image', 'low', 'le2048', 'land', 1, 15, 'resized_and_recompressed', 2048, 1366, 552, 368, False, 93.0, '4:2:0', True, -0.001102, 1.000771, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE2048_LAND_04', 'facebook', 'image', 'low', 'le2048', 'land', 1, 18, 'resized_and_recompressed', 2048, 1365, 552, 368, False, 93.0, '4:2:0', True, -0.001012, 1.000521, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE2048_PORT_01', 'facebook', 'image', 'low', 'le2048', 'port', 1, 12, 'resized_and_recompressed', 1365, 2048, 480, 720, False, 93.0, '4:2:0', True, -0.000226, 1.000053, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE2048_PORT_02', 'facebook', 'image', 'low', 'le2048', 'port', 1, 13, 'resized_and_recompressed', 1536, 2048, 480, 640, False, 93.0, '4:2:0', True, -0.000577, 1.000332, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE2048_PORT_03', 'facebook', 'image', 'low', 'le2048', 'port', 1, 16, 'resized_and_recompressed', 1152, 2048, 480, 853, False, 93.0, '4:2:0', True, -0.001285, 1.000203, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE2048_PORT_04', 'facebook', 'image', 'low', 'le2048', 'port', 1, 17, 'resized_and_recompressed', 1357, 2048, 480, 724, False, 93.0, '4:2:0', True, -0.000175, 0.999678, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE2048_PORT_05', 'facebook', 'image', 'low', 'le2048', 'port', 1, 19, 'resized_and_recompressed', 1365, 2048, 480, 720, False, 93.0, '4:2:0', True, -0.000264, 1.00008, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE2048_PORT_06', 'facebook', 'image', 'low', 'le2048', 'port', 1, 20, 'resized_and_recompressed', 1633, 2048, 480, 602, False, 93.0, '4:2:0', True, -0.000578, 1.00037, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE512_PORT_01', 'facebook', 'image', 'low', 'le512', 'port', 1, 21, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE512_PORT_02', 'facebook', 'image', 'low', 'le512', 'port', 1, 22, 'container_or_metadata_only', 444, 512, 444, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE512_PORT_03', 'facebook', 'image', 'low', 'le512', 'port', 1, 23, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE512_PORT_04', 'facebook', 'image', 'low', 'le512', 'port', 1, 24, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE512_PORT_05', 'facebook', 'image', 'low', 'le512', 'port', 1, 25, 'container_or_metadata_only', 444, 512, 444, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE512_PORT_06', 'facebook', 'image', 'low', 'le512', 'port', 1, 26, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE512_PORT_07', 'facebook', 'image', 'low', 'le512', 'port', 1, 27, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE512_PORT_08', 'facebook', 'image', 'low', 'le512', 'port', 1, 28, 'container_or_metadata_only', 444, 512, 444, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE512_PORT_09', 'facebook', 'image', 'low', 'le512', 'port', 1, 29, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_LOW_LE512_PORT_10', 'facebook', 'image', 'low', 'le512', 'port', 1, 30, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE1080_LAND_01', 'facebook', 'image', 'mid', 'le1080', 'land', 1, 4, 'resized_and_recompressed', 1080, 608, 1066, 600, False, 93.0, '4:2:0', True, 0.000278, 0.999576, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE1080_LAND_02', 'facebook', 'image', 'mid', 'le1080', 'land', 1, 5, 'resized_and_recompressed', 1080, 608, 1066, 600, False, 93.0, '4:2:0', True, 0.000331, 0.999163, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE1080_LAND_03', 'facebook', 'image', 'mid', 'le1080', 'land', 1, 6, 'resized_and_recompressed', 1080, 608, 1066, 600, False, 93.0, '4:2:0', True, 0.00044, 0.999868, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE1080_LAND_04', 'facebook', 'image', 'mid', 'le1080', 'land', 1, 7, 'resized_and_recompressed', 1080, 608, 1066, 600, False, 93.0, '4:2:0', True, -0.001398, 0.998748, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE1080_LAND_05', 'facebook', 'image', 'mid', 'le1080', 'land', 1, 8, 'resized_and_recompressed', 1080, 608, 1066, 600, False, 93.0, '4:2:0', True, -0.001345, 0.998676, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE1080_LAND_06', 'facebook', 'image', 'mid', 'le1080', 'land', 1, 9, 'container_or_metadata_only', 1080, 594, 1080, 594, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE1080_LAND_07', 'facebook', 'image', 'mid', 'le1080', 'land', 1, 11, 'resized_and_recompressed', 1080, 719, 901, 600, False, 93.0, '4:2:0', True, -0.000121, 0.999392, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE1080_PORT_01', 'facebook', 'image', 'mid', 'le1080', 'port', 1, 1, 'resized_and_recompressed', 844, 1080, 843, 1079, False, 93.0, '4:2:0', True, -7e-05, 1.00043, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE1080_PORT_02', 'facebook', 'image', 'mid', 'le1080', 'port', 1, 2, 'container_or_metadata_only', 720, 1080, 720, 1080, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE1080_PORT_03', 'facebook', 'image', 'mid', 'le1080', 'port', 1, 3, 'container_or_metadata_only', 767, 1080, 767, 1080, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE2048_LAND_01', 'facebook', 'image', 'mid', 'le2048', 'land', 1, 10, 'resized_and_recompressed', 2048, 1365, 900, 600, False, 93.0, '4:2:0', True, -0.001479, 1.000004, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE2048_LAND_02', 'facebook', 'image', 'mid', 'le2048', 'land', 1, 14, 'resized_and_recompressed', 2048, 1365, 900, 600, False, 93.0, '4:2:0', True, -0.001556, 1.000597, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE2048_LAND_03', 'facebook', 'image', 'mid', 'le2048', 'land', 1, 15, 'resized_and_recompressed', 2048, 1366, 900, 600, False, 93.0, '4:2:0', True, -0.001735, 1.000844, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE2048_LAND_04', 'facebook', 'image', 'mid', 'le2048', 'land', 1, 18, 'resized_and_recompressed', 2048, 1365, 900, 600, False, 93.0, '4:2:0', True, -0.001683, 1.000553, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE2048_PORT_01', 'facebook', 'image', 'mid', 'le2048', 'port', 1, 12, 'resized_and_recompressed', 1365, 2048, 843, 1265, False, 93.0, '4:2:0', True, -0.001208, 1.00005, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE2048_PORT_02', 'facebook', 'image', 'mid', 'le2048', 'port', 1, 13, 'resized_and_recompressed', 1536, 2048, 843, 1124, False, 93.0, '4:2:0', True, -0.001765, 1.00021, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE2048_PORT_03', 'facebook', 'image', 'mid', 'le2048', 'port', 1, 16, 'resized_and_recompressed', 1152, 2048, 843, 1499, False, 93.0, '4:2:0', True, -0.000783, 1.000219, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE2048_PORT_04', 'facebook', 'image', 'mid', 'le2048', 'port', 1, 17, 'resized_and_recompressed', 1357, 2048, 843, 1272, False, 93.0, '4:2:0', True, -0.001231, 0.999861, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE2048_PORT_05', 'facebook', 'image', 'mid', 'le2048', 'port', 1, 19, 'resized_and_recompressed', 1365, 2048, 843, 1265, False, 93.0, '4:2:0', True, -0.001334, 1.00033, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE2048_PORT_06', 'facebook', 'image', 'mid', 'le2048', 'port', 1, 20, 'resized_and_recompressed', 1633, 2048, 843, 1057, False, 93.0, '4:2:0', True, -0.002298, 1.000821, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE512_PORT_01', 'facebook', 'image', 'mid', 'le512', 'port', 1, 21, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE512_PORT_02', 'facebook', 'image', 'mid', 'le512', 'port', 1, 22, 'container_or_metadata_only', 444, 512, 444, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE512_PORT_03', 'facebook', 'image', 'mid', 'le512', 'port', 1, 23, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE512_PORT_04', 'facebook', 'image', 'mid', 'le512', 'port', 1, 24, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE512_PORT_05', 'facebook', 'image', 'mid', 'le512', 'port', 1, 25, 'container_or_metadata_only', 444, 512, 444, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE512_PORT_06', 'facebook', 'image', 'mid', 'le512', 'port', 1, 26, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE512_PORT_07', 'facebook', 'image', 'mid', 'le512', 'port', 1, 27, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE512_PORT_08', 'facebook', 'image', 'mid', 'le512', 'port', 1, 28, 'container_or_metadata_only', 444, 512, 444, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE512_PORT_09', 'facebook', 'image', 'mid', 'le512', 'port', 1, 29, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_IMAGE_MID_LE512_PORT_10', 'facebook', 'image', 'mid', 'le512', 'port', 1, 30, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('FACEBOOK_VIDEO_HIGH_1080P_LAND_01', 'facebook', 'video', 'high', '1080p', 'land', 1, 1, 'resized_and_recompressed', 1920, 1080, 1280, 720, False, None, None, None, 0.004243, 0.997855, 25.19, 0.08606, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_1080P_LAND_02', 'facebook', 'video', 'high', '1080p', 'land', 1, 2, 'resized_and_recompressed', 1920, 1080, 1280, 720, False, None, None, None, 0.004675, 0.998476, 25.63, 0.11218, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_1080P_LAND_03', 'facebook', 'video', 'high', '1080p', 'land', 1, 3, 'resized_and_recompressed', 1920, 1080, 1280, 720, False, None, None, None, 0.004176, 0.999494, 26.63, 0.06227, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_1080P_LAND_04', 'facebook', 'video', 'high', '1080p', 'land', 1, 4, 'resized_and_recompressed', 1920, 1080, 1280, 720, False, None, None, None, 0.00354, 0.999201, 25.13, 0.03722, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_1080P_LAND_05', 'facebook', 'video', 'high', '1080p', 'land', 1, 5, 'resized_and_recompressed', 1920, 1080, 1280, 720, False, None, None, None, 0.005282, 0.995415, 24.72, 0.08232, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_1080P_LAND_06', 'facebook', 'video', 'high', '1080p', 'land', 1, 13, 'resized_and_recompressed', 1920, 1080, 1280, 720, False, None, None, None, 0.003073, 0.999824, 26.0, 0.19061, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_1080P_LAND_07', 'facebook', 'video', 'high', '1080p', 'land', 1, 15, 'resized_and_recompressed', 1920, 1080, 1280, 720, False, None, None, None, 0.003866, 0.997937, 26.73, 0.15718, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_1080P_PORT_01', 'facebook', 'video', 'high', '1080p', 'port', 1, 11, 'resized_and_recompressed', 1080, 1920, 720, 1280, False, None, None, None, 0.003164, 1.000557, 26.0, 0.18336, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_1080P_PORT_02', 'facebook', 'video', 'high', '1080p', 'port', 1, 12, 'resized_and_recompressed', 1080, 1920, 720, 1280, False, None, None, None, 0.00342, 0.999283, 26.26, 0.18666, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_1080P_PORT_03', 'facebook', 'video', 'high', '1080p', 'port', 1, 14, 'resized_and_recompressed', 1080, 1920, 720, 1280, False, None, None, None, 0.003702, 0.998939, 26.0, 0.18199, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_2160P_LAND_01', 'facebook', 'video', 'high', '2160p', 'land', 1, 23, 'resized_and_recompressed', 3840, 2160, 1280, 720, False, None, None, None, 0.004367, 0.996082, 26.71, 0.08608, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_2160P_LAND_02', 'facebook', 'video', 'high', '2160p', 'land', 1, 24, 'resized_and_recompressed', 3840, 2160, 1280, 720, False, None, None, None, 0.002627, 1.000594, 25.46, 0.06753, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_2160P_LAND_03', 'facebook', 'video', 'high', '2160p', 'land', 1, 25, 'resized_and_recompressed', 3840, 2160, 1280, 720, False, None, None, None, 0.001407, 1.001281, 26.53, 0.02546, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_2160P_LAND_04', 'facebook', 'video', 'high', '2160p', 'land', 1, 27, 'resized_and_recompressed', 3840, 2160, 1280, 720, False, None, None, None, 0.003148, 0.998953, 23.67, 0.0515, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_2160P_PORT_01', 'facebook', 'video', 'high', '2160p', 'port', 1, 21, 'resized_and_recompressed', 2160, 3840, 720, 1280, False, None, None, None, 0.003691, 1.000225, 24.49, 0.05372, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_2160P_PORT_02', 'facebook', 'video', 'high', '2160p', 'port', 1, 22, 'resized_and_recompressed', 2160, 3840, 720, 1280, False, None, None, None, 0.004525, 0.997904, 25.51, 0.03513, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_2160P_PORT_03', 'facebook', 'video', 'high', '2160p', 'port', 1, 26, 'resized_and_recompressed', 2132, 3840, 708, 1276, False, None, None, None, 0.003267, 1.000961, 25.23, 0.05746, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_2160P_PORT_04', 'facebook', 'video', 'high', '2160p', 'port', 1, 28, 'resized_and_recompressed', 2160, 3840, 720, 1280, False, None, None, None, 0.004534, 0.999, 25.57, 0.06755, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_2160P_PORT_05', 'facebook', 'video', 'high', '2160p', 'port', 1, 29, 'resized_and_recompressed', 2160, 3840, 720, 1280, False, None, None, None, 0.006364, 0.995939, 25.0, 0.06217, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_2160P_PORT_06', 'facebook', 'video', 'high', '2160p', 'port', 1, 30, 'resized_and_recompressed', 2160, 3840, 720, 1280, False, None, None, None, 0.005005, 0.998817, 26.39, 0.05838, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_720P_LAND_01', 'facebook', 'video', 'high', '720p', 'land', 1, 6, 'recompressed_same_geometry', 1280, 720, 1280, 720, False, None, None, None, 0.001092, 0.999126, 25.0, 0.32984, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_720P_LAND_02', 'facebook', 'video', 'high', '720p', 'land', 1, 7, 'recompressed_same_geometry', 1280, 720, 1280, 720, False, None, None, None, 0.0007, 0.999695, 25.0, 0.35769, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_720P_LAND_03', 'facebook', 'video', 'high', '720p', 'land', 1, 8, 'recompressed_same_geometry', 1280, 720, 1280, 720, False, None, None, None, 0.000703, 0.999306, 24.8, 0.34625, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_720P_LAND_04', 'facebook', 'video', 'high', '720p', 'land', 1, 9, 'recompressed_same_geometry', 1280, 720, 1280, 720, False, None, None, None, 0.000494, 1.000939, 24.35, 0.33973, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_720P_LAND_05', 'facebook', 'video', 'high', '720p', 'land', 1, 10, 'recompressed_same_geometry', 1280, 720, 1280, 720, False, None, None, None, 0.000779, 0.999625, 26.46, 0.32849, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_720P_LAND_06', 'facebook', 'video', 'high', '720p', 'land', 1, 16, 'recompressed_same_geometry', 1280, 720, 1280, 720, False, None, None, None, 0.000674, 0.999628, 25.0, 0.42224, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_720P_LAND_07', 'facebook', 'video', 'high', '720p', 'land', 1, 19, 'recompressed_same_geometry', 1280, 720, 1280, 720, False, None, None, None, 0.001129, 0.999135, 23.5, 0.48331, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_720P_PORT_01', 'facebook', 'video', 'high', '720p', 'port', 1, 17, 'recompressed_same_geometry', 720, 1280, 720, 1280, False, None, None, None, 4.3e-05, 1.000151, 25.49, 0.3091, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_720P_PORT_02', 'facebook', 'video', 'high', '720p', 'port', 1, 18, 'recompressed_same_geometry', 720, 1280, 720, 1280, False, None, None, None, 0.000329, 0.999758, 24.47, 0.41119, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_HIGH_720P_PORT_03', 'facebook', 'video', 'high', '720p', 'port', 1, 20, 'recompressed_same_geometry', 720, 1280, 720, 1280, False, None, None, None, 0.000565, 0.999863, 25.26, 0.36243, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_1080P_LAND_01', 'facebook', 'video', 'low', '1080p', 'land', 1, 1, 'resized_and_recompressed', 1920, 1080, 640, 360, False, None, None, None, 0.004363, 0.998136, 28.0, 0.01831, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_1080P_LAND_02', 'facebook', 'video', 'low', '1080p', 'land', 1, 2, 'resized_and_recompressed', 1920, 1080, 640, 360, False, None, None, None, 0.006493, 0.996663, 29.0, 0.02776, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_1080P_LAND_03', 'facebook', 'video', 'low', '1080p', 'land', 1, 3, 'resized_and_recompressed', 1920, 1080, 640, 360, False, None, None, None, 0.00653, 0.997406, 28.4, 0.01866, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_1080P_LAND_04', 'facebook', 'video', 'low', '1080p', 'land', 1, 4, 'resized_and_recompressed', 1920, 1080, 640, 360, False, None, None, None, 0.005297, 0.996766, 28.36, 0.00753, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_1080P_LAND_05', 'facebook', 'video', 'low', '1080p', 'land', 1, 5, 'resized_and_recompressed', 1920, 1080, 640, 360, False, None, None, None, 0.006896, 0.993455, 27.5, 0.01572, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_1080P_LAND_06', 'facebook', 'video', 'low', '1080p', 'land', 1, 13, 'resized_and_recompressed', 1920, 1080, 640, 360, False, None, None, None, 0.005237, 0.998845, 29.67, 0.0393, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_1080P_LAND_07', 'facebook', 'video', 'low', '1080p', 'land', 1, 15, 'resized_and_recompressed', 1920, 1080, 640, 360, False, None, None, None, 0.004721, 0.996465, 30.2, 0.04244, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_1080P_PORT_01', 'facebook', 'video', 'low', '1080p', 'port', 1, 11, 'resized_and_recompressed', 1080, 1920, 360, 640, False, None, None, None, 0.005943, 0.998769, 30.0, 0.04243, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_1080P_PORT_02', 'facebook', 'video', 'low', '1080p', 'port', 1, 12, 'resized_and_recompressed', 1080, 1920, 360, 640, False, None, None, None, 0.006324, 0.994734, 30.0, 0.03879, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_1080P_PORT_03', 'facebook', 'video', 'low', '1080p', 'port', 1, 14, 'resized_and_recompressed', 1080, 1920, 360, 640, False, None, None, None, 0.00504, 0.99721, 29.37, 0.04204, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_2160P_LAND_01', 'facebook', 'video', 'low', '2160p', 'land', 1, 23, 'resized_and_recompressed', 3840, 2160, 640, 360, False, None, None, None, 0.007861, 0.987779, 30.72, 0.02008, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_2160P_LAND_02', 'facebook', 'video', 'low', '2160p', 'land', 1, 24, 'resized_and_recompressed', 3840, 2160, 640, 360, False, None, None, None, 0.003915, 0.998602, 29.0, 0.01246, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_2160P_LAND_03', 'facebook', 'video', 'low', '2160p', 'land', 1, 25, 'resized_and_recompressed', 3840, 2160, 640, 360, False, None, None, None, 0.003465, 0.997976, 29.71, 0.00598, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_2160P_LAND_04', 'facebook', 'video', 'low', '2160p', 'land', 1, 27, 'resized_and_recompressed', 3840, 2160, 640, 360, False, None, None, None, 0.003534, 0.998716, 27.22, 0.01266, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_2160P_PORT_01', 'facebook', 'video', 'low', '2160p', 'port', 1, 21, 'resized_and_recompressed', 2160, 3840, 360, 640, False, None, None, None, 0.006514, 0.99763, 28.37, 0.00987, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_2160P_PORT_02', 'facebook', 'video', 'low', '2160p', 'port', 1, 22, 'resized_and_recompressed', 2160, 3840, 360, 640, False, None, None, None, 0.007757, 0.994142, 28.38, 0.00704, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_2160P_PORT_03', 'facebook', 'video', 'low', '2160p', 'port', 1, 26, 'resized_and_recompressed', 2132, 3840, 352, 634, False, None, None, None, 0.003749, 1.000003, 29.0, 0.00933, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_2160P_PORT_04', 'facebook', 'video', 'low', '2160p', 'port', 1, 28, 'resized_and_recompressed', 2160, 3840, 360, 640, False, None, None, None, 0.005628, 0.99727, 29.44, 0.01544, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_2160P_PORT_05', 'facebook', 'video', 'low', '2160p', 'port', 1, 29, 'resized_and_recompressed', 2160, 3840, 360, 640, False, None, None, None, 0.00802, 0.995468, 29.0, 0.01223, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_2160P_PORT_06', 'facebook', 'video', 'low', '2160p', 'port', 1, 30, 'resized_and_recompressed', 2160, 3840, 360, 640, False, None, None, None, 0.007052, 0.996821, 29.74, 0.01362, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_720P_LAND_01', 'facebook', 'video', 'low', '720p', 'land', 1, 6, 'resized_and_recompressed', 1280, 720, 640, 360, False, None, None, None, 0.006686, 0.996837, 30.65, 0.07079, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_720P_LAND_02', 'facebook', 'video', 'low', '720p', 'land', 1, 7, 'resized_and_recompressed', 1280, 720, 640, 360, False, None, None, None, 0.004277, 0.998193, 30.0, 0.07274, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_720P_LAND_03', 'facebook', 'video', 'low', '720p', 'land', 1, 8, 'resized_and_recompressed', 1280, 720, 640, 360, False, None, None, None, 0.004734, 0.997956, 29.0, 0.07151, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_720P_LAND_04', 'facebook', 'video', 'low', '720p', 'land', 1, 9, 'resized_and_recompressed', 1280, 720, 640, 360, False, None, None, None, 0.002692, 1.005307, 29.0, 0.05935, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_720P_LAND_05', 'facebook', 'video', 'low', '720p', 'land', 1, 10, 'resized_and_recompressed', 1280, 720, 640, 360, False, None, None, None, 0.003395, 1.001936, 29.72, 0.0868, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_720P_LAND_06', 'facebook', 'video', 'low', '720p', 'land', 1, 16, 'resized_and_recompressed', 1280, 720, 640, 360, False, None, None, None, 0.004825, 0.998483, 29.54, 0.08903, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_720P_LAND_07', 'facebook', 'video', 'low', '720p', 'land', 1, 19, 'resized_and_recompressed', 1280, 720, 640, 360, False, None, None, None, 0.006492, 0.994607, 29.18, 0.08397, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_720P_PORT_01', 'facebook', 'video', 'low', '720p', 'port', 1, 17, 'resized_and_recompressed', 720, 1280, 360, 640, False, None, None, None, 0.002316, 0.999006, 27.28, 0.07855, 'h264', 'Main', 30.0, 150.0, True, -1.0),
    ('FACEBOOK_VIDEO_LOW_720P_PORT_02', 'facebook', 'video', 'low', '720p', 'port', 1, 18, 'resized_and_recompressed', 720, 1280, 360, 640, False, None, None, None, 0.004237, 0.997289, 29.0, 0.0882, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_LOW_720P_PORT_03', 'facebook', 'video', 'low', '720p', 'port', 1, 20, 'resized_and_recompressed', 720, 1280, 360, 640, False, None, None, None, 0.005795, 0.997414, 29.7, 0.07996, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_1080P_LAND_01', 'facebook', 'video', 'mid', '1080p', 'land', 1, 1, 'resized_and_recompressed', 1920, 1080, 640, 360, False, None, None, None, 0.004363, 0.998136, 28.0, 0.01831, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_1080P_LAND_02', 'facebook', 'video', 'mid', '1080p', 'land', 1, 2, 'resized_and_recompressed', 1920, 1080, 640, 360, False, None, None, None, 0.006493, 0.996663, 29.0, 0.02776, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_1080P_LAND_03', 'facebook', 'video', 'mid', '1080p', 'land', 1, 3, 'resized_and_recompressed', 1920, 1080, 640, 360, False, None, None, None, 0.00653, 0.997406, 28.4, 0.01866, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_1080P_LAND_04', 'facebook', 'video', 'mid', '1080p', 'land', 1, 4, 'resized_and_recompressed', 1920, 1080, 640, 360, False, None, None, None, 0.005297, 0.996766, 28.36, 0.00753, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_1080P_LAND_05', 'facebook', 'video', 'mid', '1080p', 'land', 1, 5, 'resized_and_recompressed', 1920, 1080, 640, 360, False, None, None, None, 0.006896, 0.993455, 27.5, 0.01572, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_1080P_LAND_06', 'facebook', 'video', 'mid', '1080p', 'land', 1, 13, 'resized_and_recompressed', 1920, 1080, 640, 360, False, None, None, None, 0.005237, 0.998845, 29.67, 0.0393, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_1080P_LAND_07', 'facebook', 'video', 'mid', '1080p', 'land', 1, 15, 'resized_and_recompressed', 1920, 1080, 640, 360, False, None, None, None, 0.004721, 0.996465, 30.2, 0.04244, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_1080P_PORT_01', 'facebook', 'video', 'mid', '1080p', 'port', 1, 11, 'resized_and_recompressed', 1080, 1920, 360, 640, False, None, None, None, 0.005943, 0.998769, 30.0, 0.04243, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_1080P_PORT_02', 'facebook', 'video', 'mid', '1080p', 'port', 1, 12, 'resized_and_recompressed', 1080, 1920, 360, 640, False, None, None, None, 0.006324, 0.994734, 30.0, 0.03879, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_1080P_PORT_03', 'facebook', 'video', 'mid', '1080p', 'port', 1, 14, 'resized_and_recompressed', 1080, 1920, 360, 640, False, None, None, None, 0.00504, 0.99721, 29.37, 0.04204, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_2160P_LAND_01', 'facebook', 'video', 'mid', '2160p', 'land', 1, 23, 'resized_and_recompressed', 3840, 2160, 640, 360, False, None, None, None, 0.007861, 0.987779, 30.72, 0.02008, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_2160P_LAND_02', 'facebook', 'video', 'mid', '2160p', 'land', 1, 24, 'resized_and_recompressed', 3840, 2160, 640, 360, False, None, None, None, 0.003915, 0.998602, 29.0, 0.01246, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_2160P_LAND_03', 'facebook', 'video', 'mid', '2160p', 'land', 1, 25, 'resized_and_recompressed', 3840, 2160, 640, 360, False, None, None, None, 0.003465, 0.997976, 29.71, 0.00598, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_2160P_LAND_04', 'facebook', 'video', 'mid', '2160p', 'land', 1, 27, 'resized_and_recompressed', 3840, 2160, 640, 360, False, None, None, None, 0.003534, 0.998716, 27.22, 0.01266, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_2160P_PORT_01', 'facebook', 'video', 'mid', '2160p', 'port', 1, 21, 'resized_and_recompressed', 2160, 3840, 360, 640, False, None, None, None, 0.006514, 0.99763, 28.37, 0.00987, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_2160P_PORT_02', 'facebook', 'video', 'mid', '2160p', 'port', 1, 22, 'resized_and_recompressed', 2160, 3840, 360, 640, False, None, None, None, 0.007757, 0.994142, 28.38, 0.00704, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_2160P_PORT_03', 'facebook', 'video', 'mid', '2160p', 'port', 1, 26, 'resized_and_recompressed', 2132, 3840, 352, 634, False, None, None, None, 0.003749, 1.000003, 29.0, 0.00933, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_2160P_PORT_04', 'facebook', 'video', 'mid', '2160p', 'port', 1, 28, 'resized_and_recompressed', 2160, 3840, 360, 640, False, None, None, None, 0.005628, 0.99727, 29.44, 0.01544, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_2160P_PORT_05', 'facebook', 'video', 'mid', '2160p', 'port', 1, 29, 'resized_and_recompressed', 2160, 3840, 360, 640, False, None, None, None, 0.00802, 0.995468, 29.0, 0.01223, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_2160P_PORT_06', 'facebook', 'video', 'mid', '2160p', 'port', 1, 30, 'resized_and_recompressed', 2160, 3840, 360, 640, False, None, None, None, 0.007052, 0.996821, 29.74, 0.01362, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_720P_LAND_01', 'facebook', 'video', 'mid', '720p', 'land', 1, 6, 'resized_and_recompressed', 1280, 720, 640, 360, False, None, None, None, 0.006686, 0.996837, 30.65, 0.07079, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_720P_LAND_02', 'facebook', 'video', 'mid', '720p', 'land', 1, 7, 'resized_and_recompressed', 1280, 720, 640, 360, False, None, None, None, 0.004277, 0.998193, 30.0, 0.07274, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_720P_LAND_03', 'facebook', 'video', 'mid', '720p', 'land', 1, 8, 'resized_and_recompressed', 1280, 720, 640, 360, False, None, None, None, 0.004734, 0.997956, 29.0, 0.07151, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_720P_LAND_04', 'facebook', 'video', 'mid', '720p', 'land', 1, 9, 'resized_and_recompressed', 1280, 720, 640, 360, False, None, None, None, 0.002692, 1.005307, 29.0, 0.05935, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_720P_LAND_05', 'facebook', 'video', 'mid', '720p', 'land', 1, 10, 'resized_and_recompressed', 1280, 720, 640, 360, False, None, None, None, 0.003395, 1.001936, 29.72, 0.0868, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_720P_LAND_06', 'facebook', 'video', 'mid', '720p', 'land', 1, 16, 'resized_and_recompressed', 1280, 720, 640, 360, False, None, None, None, 0.004825, 0.998483, 29.54, 0.08903, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_720P_LAND_07', 'facebook', 'video', 'mid', '720p', 'land', 1, 19, 'resized_and_recompressed', 1280, 720, 640, 360, False, None, None, None, 0.006492, 0.994607, 29.18, 0.08397, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_720P_PORT_01', 'facebook', 'video', 'mid', '720p', 'port', 1, 17, 'resized_and_recompressed', 720, 1280, 360, 640, False, None, None, None, 0.002316, 0.999006, 27.28, 0.07855, 'h264', 'Main', 30.0, 150.0, True, -1.0),
    ('FACEBOOK_VIDEO_MID_720P_PORT_02', 'facebook', 'video', 'mid', '720p', 'port', 1, 18, 'resized_and_recompressed', 720, 1280, 360, 640, False, None, None, None, 0.004237, 0.997289, 29.0, 0.0882, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('FACEBOOK_VIDEO_MID_720P_PORT_03', 'facebook', 'video', 'mid', '720p', 'port', 1, 20, 'resized_and_recompressed', 720, 1280, 360, 640, False, None, None, None, 0.005795, 0.997414, 29.7, 0.07996, 'h264', 'Main', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_IMAGE_SHARED_LE1080_LAND_01', 'instagram', 'image', 'shared', 'le1080', 'land', 1, 4, 'resized_and_recompressed', 1080, 608, 608, 608, True, 92.0, '4:2:0', True, -3.6e-05, 1.000043, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE1080_LAND_02', 'instagram', 'image', 'shared', 'le1080', 'land', 1, 5, 'resized_and_recompressed', 1080, 608, 608, 608, True, 92.0, '4:2:0', True, -0.002513, 0.999849, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE1080_LAND_03', 'instagram', 'image', 'shared', 'le1080', 'land', 1, 6, 'resized_and_recompressed', 1080, 608, 608, 608, True, 92.0, '4:2:0', True, -7.5e-05, 1.000098, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE1080_LAND_04', 'instagram', 'image', 'shared', 'le1080', 'land', 1, 7, 'resized_and_recompressed', 1080, 608, 608, 608, True, 92.0, '4:2:0', True, 1.4e-05, 0.999998, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE1080_LAND_05', 'instagram', 'image', 'shared', 'le1080', 'land', 1, 8, 'resized_and_recompressed', 1080, 608, 608, 608, True, 92.0, '4:2:0', True, -3e-06, 1.00002, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE1080_LAND_06', 'instagram', 'image', 'shared', 'le1080', 'land', 1, 9, 'resized_and_recompressed', 1080, 594, 594, 594, True, 92.0, '4:2:0', True, -0.000211, 1.000294, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE1080_LAND_07', 'instagram', 'image', 'shared', 'le1080', 'land', 1, 11, 'resized_and_recompressed', 1080, 719, 719, 719, True, 92.0, '4:2:0', True, -0.002753, 0.99943, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE1080_PORT_01', 'instagram', 'image', 'shared', 'le1080', 'port', 1, 1, 'resized_and_recompressed', 844, 1080, 844, 844, True, 92.0, '4:2:0', True, -3.7e-05, 0.999953, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE1080_PORT_02', 'instagram', 'image', 'shared', 'le1080', 'port', 1, 2, 'resized_and_recompressed', 720, 1080, 720, 720, True, 92.0, '4:2:0', True, -4.3e-05, 1.000109, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE1080_PORT_03', 'instagram', 'image', 'shared', 'le1080', 'port', 1, 3, 'resized_and_recompressed', 767, 1080, 767, 767, True, 92.0, '4:2:0', True, -5.3e-05, 1.000108, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE2048_LAND_01', 'instagram', 'image', 'shared', 'le2048', 'land', 1, 10, 'resized_and_recompressed', 2048, 1365, 1365, 1365, True, 92.0, '4:2:0', True, -0.002652, 0.999539, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE2048_LAND_02', 'instagram', 'image', 'shared', 'le2048', 'land', 1, 14, 'resized_and_recompressed', 2048, 1365, 1365, 1365, True, 92.0, '4:2:0', True, -4.9e-05, 1.000092, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE2048_LAND_03', 'instagram', 'image', 'shared', 'le2048', 'land', 1, 15, 'resized_and_recompressed', 2048, 1366, 1366, 1366, True, 92.0, '4:2:0', True, -5.4e-05, 1.000121, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE2048_LAND_04', 'instagram', 'image', 'shared', 'le2048', 'land', 1, 18, 'resized_and_recompressed', 2048, 1365, 1365, 1365, True, 92.0, '4:2:0', True, -0.002962, 1.000085, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE2048_PORT_01', 'instagram', 'image', 'shared', 'le2048', 'port', 1, 12, 'resized_and_recompressed', 1365, 2048, 1365, 1365, True, 92.0, '4:2:0', True, -0.002783, 0.999759, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE2048_PORT_02', 'instagram', 'image', 'shared', 'le2048', 'port', 1, 13, 'resized_and_recompressed', 1536, 2048, 1536, 1536, True, 92.0, '4:2:0', True, -1.3e-05, 1.000013, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE2048_PORT_03', 'instagram', 'image', 'shared', 'le2048', 'port', 1, 16, 'resized_and_recompressed', 1152, 2048, 1152, 1152, True, 92.0, '4:2:0', True, -3.3e-05, 1.000068, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE2048_PORT_04', 'instagram', 'image', 'shared', 'le2048', 'port', 1, 17, 'resized_and_recompressed', 1357, 2048, 1357, 1357, True, 92.0, '4:2:0', True, -0.002683, 0.999437, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE2048_PORT_05', 'instagram', 'image', 'shared', 'le2048', 'port', 1, 19, 'resized_and_recompressed', 1365, 2048, 1365, 1365, True, 92.0, '4:2:0', True, -0.002843, 0.999926, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE2048_PORT_06', 'instagram', 'image', 'shared', 'le2048', 'port', 1, 20, 'resized_and_recompressed', 1633, 2048, 1633, 1633, True, 92.0, '4:2:0', True, -0.002904, 0.999884, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE512_PORT_01', 'instagram', 'image', 'shared', 'le512', 'port', 1, 21, 'resized_and_recompressed', 443, 512, 443, 443, True, 92.0, '4:2:0', True, -0.002596, 0.999995, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE512_PORT_02', 'instagram', 'image', 'shared', 'le512', 'port', 1, 22, 'resized_and_recompressed', 444, 512, 444, 444, True, 92.0, '4:2:0', True, 3.7e-05, 0.999955, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE512_PORT_03', 'instagram', 'image', 'shared', 'le512', 'port', 1, 23, 'resized_and_recompressed', 443, 512, 443, 443, True, 92.0, '4:2:0', True, -0.002356, 0.999816, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE512_PORT_04', 'instagram', 'image', 'shared', 'le512', 'port', 1, 24, 'resized_and_recompressed', 443, 512, 443, 443, True, 92.0, '4:2:0', True, -0.002602, 1.000001, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE512_PORT_05', 'instagram', 'image', 'shared', 'le512', 'port', 1, 25, 'resized_and_recompressed', 444, 512, 444, 444, True, 92.0, '4:2:0', True, 2.7e-05, 0.999965, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE512_PORT_06', 'instagram', 'image', 'shared', 'le512', 'port', 1, 26, 'resized_and_recompressed', 443, 512, 443, 443, True, 92.0, '4:2:0', True, -0.002431, 0.999898, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE512_PORT_07', 'instagram', 'image', 'shared', 'le512', 'port', 1, 27, 'resized_and_recompressed', 443, 512, 443, 443, True, 92.0, '4:2:0', True, -0.002588, 0.999988, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE512_PORT_08', 'instagram', 'image', 'shared', 'le512', 'port', 1, 28, 'resized_and_recompressed', 444, 512, 444, 444, True, 92.0, '4:2:0', True, 4.2e-05, 0.999948, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE512_PORT_09', 'instagram', 'image', 'shared', 'le512', 'port', 1, 29, 'resized_and_recompressed', 443, 512, 443, 443, True, 92.0, '4:2:0', True, -0.002499, 0.999978, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_IMAGE_SHARED_LE512_PORT_10', 'instagram', 'image', 'shared', 'le512', 'port', 1, 30, 'resized_and_recompressed', 443, 512, 443, 443, True, 92.0, '4:2:0', True, -0.002598, 0.999999, None, None, None, None, None, None, None, 0),
    ('INSTAGRAM_VIDEO_SHARED_1080P_LAND_01', 'instagram', 'video', 'shared', '1080p', 'land', 1, 1, 'resized_and_recompressed', 1920, 1080, 720, 720, False, None, None, None, 0.005321, 0.995396, 25.24, 0.06523, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_1080P_LAND_02', 'instagram', 'video', 'shared', '1080p', 'land', 1, 2, 'resized_and_recompressed', 1920, 1080, 720, 720, False, None, None, None, 0.005145, 0.998726, 25.73, 0.07671, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_1080P_LAND_03', 'instagram', 'video', 'shared', '1080p', 'land', 1, 3, 'resized_and_recompressed', 1920, 1080, 720, 720, False, None, None, None, 0.005192, 0.998952, 26.37, 0.03998, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_1080P_LAND_04', 'instagram', 'video', 'shared', '1080p', 'land', 1, 4, 'resized_and_recompressed', 1920, 1080, 720, 720, False, None, None, None, 0.0045, 0.998368, 24.77, 0.03612, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_1080P_LAND_05', 'instagram', 'video', 'shared', '1080p', 'land', 1, 5, 'resized_and_recompressed', 1920, 1080, 720, 720, False, None, None, None, 0.006437, 0.992149, 24.73, 0.06102, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_1080P_LAND_06', 'instagram', 'video', 'shared', '1080p', 'land', 1, 13, 'resized_and_recompressed', 1920, 1080, 720, 720, False, None, None, None, 0.00343, 1.000168, 26.0, 0.11773, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_1080P_LAND_07', 'instagram', 'video', 'shared', '1080p', 'land', 1, 15, 'resized_and_recompressed', 1920, 1080, 720, 720, False, None, None, None, 0.004342, 0.997748, 27.0, 0.11126, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_1080P_PORT_01', 'instagram', 'video', 'shared', '1080p', 'port', 1, 11, 'resized_and_recompressed', 1080, 1920, 720, 720, False, None, None, None, 0.003807, 0.999583, 26.0, 0.10651, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_1080P_PORT_02', 'instagram', 'video', 'shared', '1080p', 'port', 1, 12, 'resized_and_recompressed', 1080, 1920, 720, 720, False, None, None, None, 0.00363, 0.999256, 26.25, 0.12955, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_1080P_PORT_03', 'instagram', 'video', 'shared', '1080p', 'port', 1, 14, 'resized_and_recompressed', 1080, 1920, 720, 720, False, None, None, None, 0.004112, 0.99881, 26.35, 0.11374, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_2160P_LAND_01', 'instagram', 'video', 'shared', '2160p', 'land', 1, 23, 'resized_and_recompressed', 3840, 2160, 720, 720, False, None, None, None, 0.004516, 0.996493, 27.0, 0.04614, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_2160P_LAND_02', 'instagram', 'video', 'shared', '2160p', 'land', 1, 24, 'resized_and_recompressed', 3840, 2160, 720, 720, False, None, None, None, 0.00269, 1.001927, 25.37, 0.03603, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_2160P_LAND_03', 'instagram', 'video', 'shared', '2160p', 'land', 1, 25, 'resized_and_recompressed', 3840, 2160, 720, 720, False, None, None, None, 0.002615, 0.999907, 26.29, 0.01904, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_2160P_LAND_04', 'instagram', 'video', 'shared', '2160p', 'land', 1, 27, 'resized_and_recompressed', 3840, 2160, 720, 720, False, None, None, None, 0.003199, 0.999084, 24.0, 0.0307, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_2160P_PORT_01', 'instagram', 'video', 'shared', '2160p', 'port', 1, 21, 'resized_and_recompressed', 2160, 3840, 720, 720, False, None, None, None, 0.004667, 0.999081, 24.54, 0.02822, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_2160P_PORT_02', 'instagram', 'video', 'shared', '2160p', 'port', 1, 22, 'resized_and_recompressed', 2160, 3840, 720, 720, False, None, None, None, 0.004827, 0.997787, 26.13, 0.02059, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_2160P_PORT_03', 'instagram', 'video', 'shared', '2160p', 'port', 1, 26, 'resized_and_recompressed', 2132, 3840, 720, 720, False, None, None, None, 0.002747, 1.001272, 25.42, 0.03622, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_2160P_PORT_04', 'instagram', 'video', 'shared', '2160p', 'port', 1, 28, 'resized_and_recompressed', 2160, 3840, 720, 720, False, None, None, None, 0.004629, 0.998627, 25.74, 0.04267, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_2160P_PORT_05', 'instagram', 'video', 'shared', '2160p', 'port', 1, 29, 'resized_and_recompressed', 2160, 3840, 720, 720, False, None, None, None, 0.00694, 0.994101, 25.0, 0.0332, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_2160P_PORT_06', 'instagram', 'video', 'shared', '2160p', 'port', 1, 30, 'resized_and_recompressed', 2160, 3840, 720, 720, False, None, None, None, 0.005049, 0.999044, 26.63, 0.0353, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_720P_LAND_01', 'instagram', 'video', 'shared', '720p', 'land', 1, 6, 'resized_and_recompressed', 1280, 720, 720, 720, False, None, None, None, 0.001835, 0.999281, 24.5, 0.2441, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_720P_LAND_02', 'instagram', 'video', 'shared', '720p', 'land', 1, 7, 'resized_and_recompressed', 1280, 720, 720, 720, False, None, None, None, 0.000967, 0.999626, 25.0, 0.2474, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_720P_LAND_03', 'instagram', 'video', 'shared', '720p', 'land', 1, 8, 'resized_and_recompressed', 1280, 720, 720, 720, False, None, None, None, 0.001093, 0.999051, 24.7, 0.26796, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_720P_LAND_04', 'instagram', 'video', 'shared', '720p', 'land', 1, 9, 'resized_and_recompressed', 1280, 720, 720, 720, False, None, None, None, 0.000621, 1.001734, 24.49, 0.19952, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_720P_LAND_05', 'instagram', 'video', 'shared', '720p', 'land', 1, 10, 'resized_and_recompressed', 1280, 720, 720, 720, False, None, None, None, 0.001125, 0.999769, 26.0, 0.19472, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_720P_LAND_06', 'instagram', 'video', 'shared', '720p', 'land', 1, 16, 'resized_and_recompressed', 1280, 720, 720, 720, False, None, None, None, 0.001138, 0.999147, 25.27, 0.28167, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_720P_LAND_07', 'instagram', 'video', 'shared', '720p', 'land', 1, 19, 'resized_and_recompressed', 1280, 720, 720, 720, False, None, None, None, 0.001422, 0.998116, 24.0, 0.27667, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_720P_PORT_01', 'instagram', 'video', 'shared', '720p', 'port', 1, 17, 'resized_and_recompressed', 720, 1280, 720, 720, False, None, None, None, 0.000218, 1.000173, 26.0, 0.22308, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_720P_PORT_02', 'instagram', 'video', 'shared', '720p', 'port', 1, 18, 'resized_and_recompressed', 720, 1280, 720, 720, False, None, None, None, 0.00083, 0.999512, 25.0, 0.24187, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('INSTAGRAM_VIDEO_SHARED_720P_PORT_03', 'instagram', 'video', 'shared', '720p', 'port', 1, 20, 'resized_and_recompressed', 720, 1280, 720, 720, False, None, None, None, 0.000618, 0.999798, 25.51, 0.22879, 'h264', 'High', 30.0, 150.0, True, 0.0),
    ('KAKAO_IMAGE_LOW_LE1080_LAND_01', 'kakao', 'image', 'low', 'le1080', 'land', 1, 4, 'resized_and_recompressed', 1080, 608, 960, 541, False, 86.0, '4:2:0', False, -0.00356, 1.003752, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE1080_LAND_02', 'kakao', 'image', 'low', 'le1080', 'land', 1, 5, 'resized_and_recompressed', 1080, 608, 960, 541, False, 86.0, '4:2:0', False, -0.004558, 1.00112, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE1080_LAND_03', 'kakao', 'image', 'low', 'le1080', 'land', 1, 6, 'resized_and_recompressed', 1080, 608, 960, 541, False, 86.0, '4:2:0', False, -0.001915, 0.99854, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE1080_LAND_04', 'kakao', 'image', 'low', 'le1080', 'land', 1, 7, 'resized_and_recompressed', 1080, 608, 960, 541, False, 86.0, '4:2:0', False, -0.002912, 0.99851, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE1080_LAND_05', 'kakao', 'image', 'low', 'le1080', 'land', 1, 8, 'resized_and_recompressed', 1080, 608, 960, 541, False, 86.0, '4:2:0', False, -0.002911, 0.998495, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE1080_LAND_06', 'kakao', 'image', 'low', 'le1080', 'land', 1, 9, 'resized_and_recompressed', 1080, 594, 960, 528, False, 86.0, '4:2:0', False, -0.002356, 1.000677, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE1080_LAND_07', 'kakao', 'image', 'low', 'le1080', 'land', 1, 11, 'resized_and_recompressed', 1080, 719, 960, 640, False, 86.0, '4:2:0', False, -0.006468, 1.004861, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE1080_PORT_01', 'kakao', 'image', 'low', 'le1080', 'port', 1, 1, 'resized_and_recompressed', 844, 1080, 751, 960, False, 86.0, '4:2:0', False, -0.001474, 0.999176, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE1080_PORT_02', 'kakao', 'image', 'low', 'le1080', 'port', 1, 2, 'resized_and_recompressed', 720, 1080, 640, 960, False, 86.0, '4:2:0', False, -0.005319, 1.002254, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE1080_PORT_03', 'kakao', 'image', 'low', 'le1080', 'port', 1, 3, 'resized_and_recompressed', 767, 1080, 682, 960, False, 86.0, '4:2:0', False, -0.002514, 1.001218, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE2048_LAND_01', 'kakao', 'image', 'low', 'le2048', 'land', 1, 10, 'resized_and_recompressed', 2048, 1365, 960, 640, False, 86.0, '4:2:0', False, -0.004826, 0.998875, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE2048_LAND_02', 'kakao', 'image', 'low', 'le2048', 'land', 1, 14, 'resized_and_recompressed', 2048, 1365, 960, 640, False, 86.0, '4:2:0', False, -0.003494, 0.999956, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE2048_LAND_03', 'kakao', 'image', 'low', 'le2048', 'land', 1, 15, 'resized_and_recompressed', 2048, 1366, 960, 641, False, 86.0, '4:2:0', False, -0.00599, 1.001358, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE2048_LAND_04', 'kakao', 'image', 'low', 'le2048', 'land', 1, 18, 'resized_and_recompressed', 2048, 1365, 960, 640, False, 86.0, '4:2:0', False, -0.005789, 1.000843, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE2048_PORT_01', 'kakao', 'image', 'low', 'le2048', 'port', 1, 12, 'resized_and_recompressed', 1365, 2048, 640, 960, False, 86.0, '4:2:0', False, -0.003693, 1.000222, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE2048_PORT_02', 'kakao', 'image', 'low', 'le2048', 'port', 1, 13, 'resized_and_recompressed', 1536, 2048, 720, 960, False, 86.0, '4:2:0', False, -0.004107, 1.000947, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE2048_PORT_03', 'kakao', 'image', 'low', 'le2048', 'port', 1, 16, 'resized_and_recompressed', 1152, 2048, 540, 960, False, 86.0, '4:2:0', False, -0.004431, 1.000785, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE2048_PORT_04', 'kakao', 'image', 'low', 'le2048', 'port', 1, 17, 'resized_and_recompressed', 1357, 2048, 637, 960, False, 86.0, '4:2:0', False, -0.004304, 1.001259, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE2048_PORT_05', 'kakao', 'image', 'low', 'le2048', 'port', 1, 19, 'resized_and_recompressed', 1365, 2048, 640, 960, False, 86.0, '4:2:0', False, -0.00458, 1.001774, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE2048_PORT_06', 'kakao', 'image', 'low', 'le2048', 'port', 1, 20, 'resized_and_recompressed', 1633, 2048, 766, 960, False, 86.0, '4:2:0', False, -0.004119, 1.001165, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE512_PORT_01', 'kakao', 'image', 'low', 'le512', 'port', 1, 21, 'recompressed_same_geometry', 443, 512, 443, 512, False, 86.0, '4:2:0', False, -0.00097, 0.999534, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE512_PORT_02', 'kakao', 'image', 'low', 'le512', 'port', 1, 22, 'recompressed_same_geometry', 444, 512, 444, 512, False, 86.0, '4:2:0', False, -0.001047, 0.999625, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE512_PORT_03', 'kakao', 'image', 'low', 'le512', 'port', 1, 23, 'recompressed_same_geometry', 443, 512, 443, 512, False, 86.0, '4:2:0', False, -0.001016, 0.998902, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE512_PORT_04', 'kakao', 'image', 'low', 'le512', 'port', 1, 24, 'recompressed_same_geometry', 443, 512, 443, 512, False, 86.0, '4:2:0', False, -0.000969, 0.999525, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE512_PORT_05', 'kakao', 'image', 'low', 'le512', 'port', 1, 25, 'recompressed_same_geometry', 444, 512, 444, 512, False, 86.0, '4:2:0', False, -0.001048, 0.999623, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE512_PORT_06', 'kakao', 'image', 'low', 'le512', 'port', 1, 26, 'recompressed_same_geometry', 443, 512, 443, 512, False, 86.0, '4:2:0', False, -0.000944, 0.998826, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE512_PORT_07', 'kakao', 'image', 'low', 'le512', 'port', 1, 27, 'recompressed_same_geometry', 443, 512, 443, 512, False, 86.0, '4:2:0', False, -0.000943, 0.999491, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE512_PORT_08', 'kakao', 'image', 'low', 'le512', 'port', 1, 28, 'recompressed_same_geometry', 444, 512, 444, 512, False, 86.0, '4:2:0', False, -0.001057, 0.999638, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE512_PORT_09', 'kakao', 'image', 'low', 'le512', 'port', 1, 29, 'recompressed_same_geometry', 443, 512, 443, 512, False, 86.0, '4:2:0', False, -0.000851, 0.998712, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_LOW_LE512_PORT_10', 'kakao', 'image', 'low', 'le512', 'port', 1, 30, 'recompressed_same_geometry', 443, 512, 443, 512, False, 86.0, '4:2:0', False, -0.000973, 0.999536, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE1080_LAND_01', 'kakao', 'image', 'normal', 'le1080', 'land', 1, 4, 'recompressed_same_geometry', 1080, 608, 1080, 608, False, 96.0, '4:2:0', False, -0.003202, 1.003085, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE1080_LAND_02', 'kakao', 'image', 'normal', 'le1080', 'land', 1, 5, 'recompressed_same_geometry', 1080, 608, 1080, 608, False, 96.0, '4:2:0', False, -0.001978, 1.000349, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE1080_LAND_03', 'kakao', 'image', 'normal', 'le1080', 'land', 1, 6, 'recompressed_same_geometry', 1080, 608, 1080, 608, False, 96.0, '4:2:0', False, -0.001678, 0.999924, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE1080_LAND_04', 'kakao', 'image', 'normal', 'le1080', 'land', 1, 7, 'recompressed_same_geometry', 1080, 608, 1080, 608, False, 96.0, '4:2:0', False, -0.00106, 0.998993, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE1080_LAND_05', 'kakao', 'image', 'normal', 'le1080', 'land', 1, 8, 'recompressed_same_geometry', 1080, 608, 1080, 608, False, 96.0, '4:2:0', False, -0.001055, 0.99899, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE1080_LAND_06', 'kakao', 'image', 'normal', 'le1080', 'land', 1, 9, 'recompressed_same_geometry', 1080, 594, 1080, 594, False, 96.0, '4:2:0', False, -0.001885, 1.000332, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE1080_LAND_07', 'kakao', 'image', 'normal', 'le1080', 'land', 1, 11, 'recompressed_same_geometry', 1080, 719, 1080, 719, False, 96.0, '4:2:0', False, -0.001919, 1.000421, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE1080_PORT_01', 'kakao', 'image', 'normal', 'le1080', 'port', 1, 1, 'recompressed_same_geometry', 844, 1080, 844, 1080, False, 96.0, '4:2:0', False, -0.001127, 0.998892, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE1080_PORT_02', 'kakao', 'image', 'normal', 'le1080', 'port', 1, 2, 'recompressed_same_geometry', 720, 1080, 720, 1080, False, 96.0, '4:2:0', False, -0.001866, 1.000278, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE1080_PORT_03', 'kakao', 'image', 'normal', 'le1080', 'port', 1, 3, 'recompressed_same_geometry', 767, 1080, 767, 1080, False, 96.0, '4:2:0', False, -0.002071, 1.000719, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE2048_LAND_01', 'kakao', 'image', 'normal', 'le2048', 'land', 1, 10, 'resized_and_recompressed', 2048, 1365, 1440, 960, False, 96.0, '4:2:0', False, -0.004199, 0.999186, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE2048_LAND_02', 'kakao', 'image', 'normal', 'le2048', 'land', 1, 14, 'resized_and_recompressed', 2048, 1365, 1440, 960, False, 96.0, '4:2:0', False, -0.00262, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE2048_LAND_03', 'kakao', 'image', 'normal', 'le2048', 'land', 1, 15, 'resized_and_recompressed', 2048, 1366, 1440, 961, False, 96.0, '4:2:0', False, -0.002741, 1.000578, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE2048_LAND_04', 'kakao', 'image', 'normal', 'le2048', 'land', 1, 18, 'resized_and_recompressed', 2048, 1365, 1440, 960, False, 96.0, '4:2:0', False, -0.004756, 1.000511, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE2048_PORT_01', 'kakao', 'image', 'normal', 'le2048', 'port', 1, 12, 'resized_and_recompressed', 1365, 2048, 960, 1440, False, 96.0, '4:2:0', False, -0.002599, 0.999789, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE2048_PORT_02', 'kakao', 'image', 'normal', 'le2048', 'port', 1, 13, 'resized_and_recompressed', 1536, 2048, 1080, 1440, False, 96.0, '4:2:0', False, -0.002759, 1.000487, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE2048_PORT_03', 'kakao', 'image', 'normal', 'le2048', 'port', 1, 16, 'resized_and_recompressed', 1152, 2048, 810, 1440, False, 96.0, '4:2:0', False, -0.002861, 1.000587, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE2048_PORT_04', 'kakao', 'image', 'normal', 'le2048', 'port', 1, 17, 'resized_and_recompressed', 1357, 2048, 955, 1440, False, 96.0, '4:2:0', False, -0.003148, 1.000734, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE2048_PORT_05', 'kakao', 'image', 'normal', 'le2048', 'port', 1, 19, 'resized_and_recompressed', 1365, 2048, 960, 1440, False, 96.0, '4:2:0', False, -0.003425, 1.001309, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE2048_PORT_06', 'kakao', 'image', 'normal', 'le2048', 'port', 1, 20, 'resized_and_recompressed', 1633, 2048, 1148, 1440, False, 96.0, '4:2:0', False, -0.003328, 1.001233, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE512_PORT_01', 'kakao', 'image', 'normal', 'le512', 'port', 1, 21, 'recompressed_same_geometry', 443, 512, 443, 512, False, 96.0, '4:2:0', False, -0.001098, 0.999782, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE512_PORT_02', 'kakao', 'image', 'normal', 'le512', 'port', 1, 22, 'recompressed_same_geometry', 444, 512, 444, 512, False, 96.0, '4:2:0', False, -0.001111, 0.999764, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE512_PORT_03', 'kakao', 'image', 'normal', 'le512', 'port', 1, 23, 'recompressed_same_geometry', 443, 512, 443, 512, False, 96.0, '4:2:0', False, -0.000929, 0.998921, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE512_PORT_04', 'kakao', 'image', 'normal', 'le512', 'port', 1, 24, 'recompressed_same_geometry', 443, 512, 443, 512, False, 96.0, '4:2:0', False, -0.001116, 0.999798, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE512_PORT_05', 'kakao', 'image', 'normal', 'le512', 'port', 1, 25, 'recompressed_same_geometry', 444, 512, 444, 512, False, 96.0, '4:2:0', False, -0.00111, 0.999761, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE512_PORT_06', 'kakao', 'image', 'normal', 'le512', 'port', 1, 26, 'recompressed_same_geometry', 443, 512, 443, 512, False, 96.0, '4:2:0', False, -0.000845, 0.998757, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE512_PORT_07', 'kakao', 'image', 'normal', 'le512', 'port', 1, 27, 'recompressed_same_geometry', 443, 512, 443, 512, False, 96.0, '4:2:0', False, -0.001096, 0.999774, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE512_PORT_08', 'kakao', 'image', 'normal', 'le512', 'port', 1, 28, 'recompressed_same_geometry', 444, 512, 444, 512, False, 96.0, '4:2:0', False, -0.001116, 0.999773, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE512_PORT_09', 'kakao', 'image', 'normal', 'le512', 'port', 1, 29, 'recompressed_same_geometry', 443, 512, 443, 512, False, 96.0, '4:2:0', False, -0.000833, 0.998775, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_NORMAL_LE512_PORT_10', 'kakao', 'image', 'normal', 'le512', 'port', 1, 30, 'recompressed_same_geometry', 443, 512, 443, 512, False, 96.0, '4:2:0', False, -0.001112, 0.999794, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE1080_LAND_01', 'kakao', 'image', 'original', 'le1080', 'land', 1, 4, 'container_or_metadata_only', 1080, 608, 1080, 608, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE1080_LAND_02', 'kakao', 'image', 'original', 'le1080', 'land', 1, 5, 'container_or_metadata_only', 1080, 608, 1080, 608, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE1080_LAND_03', 'kakao', 'image', 'original', 'le1080', 'land', 1, 6, 'container_or_metadata_only', 1080, 608, 1080, 608, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE1080_LAND_04', 'kakao', 'image', 'original', 'le1080', 'land', 1, 7, 'container_or_metadata_only', 1080, 608, 1080, 608, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE1080_LAND_05', 'kakao', 'image', 'original', 'le1080', 'land', 1, 8, 'container_or_metadata_only', 1080, 608, 1080, 608, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE1080_LAND_06', 'kakao', 'image', 'original', 'le1080', 'land', 1, 9, 'container_or_metadata_only', 1080, 594, 1080, 594, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE1080_LAND_07', 'kakao', 'image', 'original', 'le1080', 'land', 1, 11, 'container_or_metadata_only', 1080, 719, 1080, 719, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE1080_PORT_01', 'kakao', 'image', 'original', 'le1080', 'port', 1, 1, 'container_or_metadata_only', 844, 1080, 844, 1080, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE1080_PORT_02', 'kakao', 'image', 'original', 'le1080', 'port', 1, 2, 'container_or_metadata_only', 720, 1080, 720, 1080, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE1080_PORT_03', 'kakao', 'image', 'original', 'le1080', 'port', 1, 3, 'container_or_metadata_only', 767, 1080, 767, 1080, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE2048_LAND_01', 'kakao', 'image', 'original', 'le2048', 'land', 1, 10, 'container_or_metadata_only', 2048, 1365, 2048, 1365, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE2048_LAND_02', 'kakao', 'image', 'original', 'le2048', 'land', 1, 14, 'container_or_metadata_only', 2048, 1365, 2048, 1365, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE2048_LAND_03', 'kakao', 'image', 'original', 'le2048', 'land', 1, 15, 'container_or_metadata_only', 2048, 1366, 2048, 1366, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE2048_LAND_04', 'kakao', 'image', 'original', 'le2048', 'land', 1, 18, 'container_or_metadata_only', 2048, 1365, 2048, 1365, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE2048_PORT_01', 'kakao', 'image', 'original', 'le2048', 'port', 1, 12, 'container_or_metadata_only', 1365, 2048, 1365, 2048, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE2048_PORT_02', 'kakao', 'image', 'original', 'le2048', 'port', 1, 13, 'container_or_metadata_only', 1536, 2048, 1536, 2048, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE2048_PORT_03', 'kakao', 'image', 'original', 'le2048', 'port', 1, 16, 'container_or_metadata_only', 1152, 2048, 1152, 2048, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE2048_PORT_04', 'kakao', 'image', 'original', 'le2048', 'port', 1, 17, 'container_or_metadata_only', 1357, 2048, 1357, 2048, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE2048_PORT_05', 'kakao', 'image', 'original', 'le2048', 'port', 1, 19, 'container_or_metadata_only', 1365, 2048, 1365, 2048, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE2048_PORT_06', 'kakao', 'image', 'original', 'le2048', 'port', 1, 20, 'container_or_metadata_only', 1633, 2048, 1633, 2048, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE512_PORT_01', 'kakao', 'image', 'original', 'le512', 'port', 1, 21, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE512_PORT_02', 'kakao', 'image', 'original', 'le512', 'port', 1, 22, 'container_or_metadata_only', 444, 512, 444, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE512_PORT_03', 'kakao', 'image', 'original', 'le512', 'port', 1, 23, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE512_PORT_04', 'kakao', 'image', 'original', 'le512', 'port', 1, 24, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE512_PORT_05', 'kakao', 'image', 'original', 'le512', 'port', 1, 25, 'container_or_metadata_only', 444, 512, 444, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE512_PORT_06', 'kakao', 'image', 'original', 'le512', 'port', 1, 26, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE512_PORT_07', 'kakao', 'image', 'original', 'le512', 'port', 1, 27, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE512_PORT_08', 'kakao', 'image', 'original', 'le512', 'port', 1, 28, 'container_or_metadata_only', 444, 512, 444, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE512_PORT_09', 'kakao', 'image', 'original', 'le512', 'port', 1, 29, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_IMAGE_ORIGINAL_LE512_PORT_10', 'kakao', 'image', 'original', 'le512', 'port', 1, 30, 'container_or_metadata_only', 443, 512, 443, 512, False, 98.0, '4:4:4', False, 0.0, 1.0, None, None, None, None, None, None, None, 0),
    ('KAKAO_VIDEO_SINGLE_1080P_LAND_01', 'kakao', 'video', 'single', '1080p', 'land', 1, 1, 'transcoded_different_format', 1920, 1080, 1920, 1080, False, None, None, None, -8e-06, 0.999916, 20.87, 0.41141, 'hevc', 'Main', 30.0, 28.0, True, 1.0),
    ('KAKAO_VIDEO_SINGLE_1080P_LAND_02', 'kakao', 'video', 'single', '1080p', 'land', 1, 2, 'transcoded_different_format', 1920, 1080, 1920, 1080, False, None, None, None, 0.00019, 0.999378, 23.77, 0.28322, 'hevc', 'Main', 30.0, 28.0, True, 1.0),
    ('KAKAO_VIDEO_SINGLE_1080P_LAND_03', 'kakao', 'video', 'single', '1080p', 'land', 1, 3, 'transcoded_different_format', 1920, 1080, 1920, 1080, False, None, None, None, 4.1e-05, 0.999995, 21.66, 0.30574, 'hevc', 'Main', 30.0, 28.0, True, 1.0),
    ('KAKAO_VIDEO_SINGLE_1080P_LAND_04', 'kakao', 'video', 'single', '1080p', 'land', 1, 4, 'transcoded_different_format', 1920, 1080, 1920, 1080, False, None, None, None, 0.000109, 0.999948, 21.0, 0.24793, 'hevc', 'Main', 30.0, 28.0, True, 1.0),
    ('KAKAO_VIDEO_SINGLE_1080P_LAND_05', 'kakao', 'video', 'single', '1080p', 'land', 1, 5, 'transcoded_different_format', 1920, 1080, 1920, 1080, False, None, None, None, 0.000293, 0.99953, 20.55, 0.43322, 'hevc', 'Main', 30.0, 28.0, True, 1.0),
    ('KAKAO_VIDEO_SINGLE_1080P_LAND_06', 'kakao', 'video', 'single', '1080p', 'land', 1, 13, 'transcoded_different_format', 1920, 1080, 1920, 1080, False, None, None, None, 0.000221, 0.999612, 26.0, 0.36636, 'hevc', 'Main', 30.0, 28.0, True, 1.0),
    ('KAKAO_VIDEO_SINGLE_1080P_LAND_07', 'kakao', 'video', 'single', '1080p', 'land', 1, 15, 'transcoded_different_format', 1920, 1080, 1920, 1080, False, None, None, None, 0.000164, 0.999786, 19.0, 0.70773, 'hevc', 'Main', 30.0, 28.0, True, 1.0),
    ('KAKAO_VIDEO_SINGLE_1080P_PORT_01', 'kakao', 'video', 'single', '1080p', 'port', 1, 11, 'transcoded_different_format', 1080, 1920, 1080, 1920, False, None, None, None, 0.00019, 0.999766, 27.0, 0.30559, 'hevc', 'Main', 30.0, 28.0, True, 1.0),
    ('KAKAO_VIDEO_SINGLE_1080P_PORT_02', 'kakao', 'video', 'single', '1080p', 'port', 1, 12, 'transcoded_different_format', 1080, 1920, 1080, 1920, False, None, None, None, 0.000808, 0.998186, 29.69, 0.25554, 'hevc', 'Main', 30.0, 28.0, True, 1.0),
    ('KAKAO_VIDEO_SINGLE_1080P_PORT_03', 'kakao', 'video', 'single', '1080p', 'port', 1, 14, 'transcoded_different_format', 1080, 1920, 1080, 1920, False, None, None, None, 0.000388, 0.998577, 29.0, 0.24341, 'hevc', 'Main', 30.0, 28.0, True, 1.0),
    ('KAKAO_VIDEO_SINGLE_2160P_LAND_01', 'kakao', 'video', 'single', '2160p', 'land', 1, 23, 'transcoded_different_format', 3840, 2160, 1920, 1080, False, None, None, None, 0.006093, 0.989946, 36.25, 0.05371, 'hevc', 'Main', 30.0, 28.0, True, 1.0),
    ('KAKAO_VIDEO_SINGLE_2160P_LAND_02', 'kakao', 'video', 'single', '2160p', 'land', 1, 24, 'transcoded_different_format', 3840, 2160, 1920, 1080, False, None, None, None, 0.000724, 1.001226, 29.0, 0.08385, 'hevc', 'Main', 30.0, 28.0, True, 1.0),
    ('KAKAO_VIDEO_SINGLE_2160P_LAND_03', 'kakao', 'video', 'single', '2160p', 'land', 1, 25, 'transcoded_different_format', 3840, 2160, 1920, 1080, False, None, None, None, 0.001001, 1.002763, 19.33, 0.17295, 'hevc', 'Main', 30.0, 28.0, True, 1.0),
    ('KAKAO_VIDEO_SINGLE_2160P_LAND_04', 'kakao', 'video', 'single', '2160p', 'land', 1, 27, 'transcoded_different_format', 3840, 2160, 1920, 1080, False, None, None, None, 0.003082, 0.999061, 18.42, 0.21047, 'hevc', 'Main', 30.0, 28.0, True, 1.0),
    ('KAKAO_VIDEO_SINGLE_2160P_PORT_01', 'kakao', 'video', 'single', '2160p', 'port', 1, 21, 'transcoded_different_format', 2160, 3840, 1080, 1920, False, None, None, None, 0.001167, 1.001657, 25.54, 0.09074, 'hevc', 'Main', 30.0, 28.0, True, 1.0),
    ('KAKAO_VIDEO_SINGLE_2160P_PORT_02', 'kakao', 'video', 'single', '2160p', 'port', 1, 22, 'transcoded_different_format', 2160, 3840, 1080, 1920, False, None, None, None, 0.003317, 0.998621, 24.31, 0.09907, 'hevc', 'Main', 30.0, 28.0, True, 1.0),
    ('KAKAO_VIDEO_SINGLE_2160P_PORT_03', 'kakao', 'video', 'single', '2160p', 'port', 1, 26, 'transcoded_different_format', 2132, 3840, 1064, 1920, False, None, None, None, 0.003032, 0.999415, 26.19, 0.09767, 'hevc', 'Main', 30.0, 28.0, True, 1.0),
    ('KAKAO_VIDEO_SINGLE_2160P_PORT_04', 'kakao', 'video', 'single', '2160p', 'port', 1, 28, 'transcoded_different_format', 2160, 3840, 1080, 1920, False, None, None, None, 0.003477, 0.998246, 28.53, 0.08962, 'hevc', 'Main', 30.0, 28.0, True, 1.0),
    ('KAKAO_VIDEO_SINGLE_2160P_PORT_05', 'kakao', 'video', 'single', '2160p', 'port', 1, 29, 'transcoded_different_format', 2160, 3840, 1080, 1920, False, None, None, None, 0.00413, 0.996969, 25.64, 0.10673, 'hevc', 'Main', 30.0, 28.0, True, 1.0),
    ('KAKAO_VIDEO_SINGLE_2160P_PORT_06', 'kakao', 'video', 'single', '2160p', 'port', 1, 30, 'transcoded_different_format', 2160, 3840, 1080, 1920, False, None, None, None, 0.003037, 0.999196, 26.61, 0.10903, 'hevc', 'Main', 30.0, 28.0, True, 1.0),
    ('KAKAO_VIDEO_SINGLE_720P_LAND_01', 'kakao', 'video', 'single', '720p', 'land', 1, 6, 'recompressed_same_geometry', 1280, 720, 1280, 720, False, None, None, None, -0.00941, 1.015213, 23.44, 0.37548, 'h264', 'Constrained Baseline', 24.0, 24.0, False, 3.0),
    ('KAKAO_VIDEO_SINGLE_720P_LAND_02', 'kakao', 'video', 'single', '720p', 'land', 1, 7, 'recompressed_same_geometry', 1280, 720, 1280, 720, False, None, None, None, 0.000504, 0.989268, 24.0, 0.35718, 'h264', 'Constrained Baseline', 24.0, 24.0, False, 4.0),
    ('KAKAO_VIDEO_SINGLE_720P_LAND_03', 'kakao', 'video', 'single', '720p', 'land', 1, 8, 'recompressed_same_geometry', 1280, 720, 1280, 720, False, None, None, None, 0.003907, 0.996904, 23.0, 0.39432, 'h264', 'Constrained Baseline', 24.0, 24.0, False, 3.0),
    ('KAKAO_VIDEO_SINGLE_720P_LAND_04', 'kakao', 'video', 'single', '720p', 'land', 1, 9, 'recompressed_same_geometry', 1280, 720, 1280, 720, False, None, None, None, -5.4e-05, 0.986023, 25.6, 0.26518, 'h264', 'Constrained Baseline', 24.0, 22.5, False, 5.0),
    ('KAKAO_VIDEO_SINGLE_720P_LAND_05', 'kakao', 'video', 'single', '720p', 'land', 1, 10, 'recompressed_same_geometry', 1280, 720, 1280, 720, False, None, None, None, -0.004295, 1.005762, 21.39, 0.48737, 'h264', 'Constrained Baseline', 24.0, 24.0, False, 5.0),
    ('KAKAO_VIDEO_SINGLE_720P_LAND_06', 'kakao', 'video', 'single', '720p', 'land', 1, 16, 'recompressed_same_geometry', 1280, 720, 1280, 720, False, None, None, None, 0.017141, 0.961022, 27.79, 0.30343, 'h264', 'Constrained Baseline', 24.0, 20.636, False, 4.0),
    ('KAKAO_VIDEO_SINGLE_720P_LAND_07', 'kakao', 'video', 'single', '720p', 'land', 1, 19, 'recompressed_same_geometry', 1280, 720, 1280, 720, False, None, None, None, 0.012655, 0.986719, 30.0, 0.25453, 'h264', 'Constrained Baseline', 24.0, 18.25, False, 5.0),
    ('KAKAO_VIDEO_SINGLE_720P_PORT_01', 'kakao', 'video', 'single', '720p', 'port', 1, 17, 'resized_and_recompressed', 720, 1280, 406, 720, False, None, None, None, -0.026805, 1.028774, 18.0, 0.76684, 'h264', 'Constrained Baseline', 24.0, 24.0, False, 5.0),
    ('KAKAO_VIDEO_SINGLE_720P_PORT_02', 'kakao', 'video', 'single', '720p', 'port', 1, 18, 'resized_and_recompressed', 720, 1280, 406, 720, False, None, None, None, 0.006375, 0.991574, 19.36, 0.28631, 'h264', 'Constrained Baseline', 24.0, 20.0, False, 4.0),
    ('KAKAO_VIDEO_SINGLE_720P_PORT_03', 'kakao', 'video', 'single', '720p', 'port', 1, 20, 'resized_and_recompressed', 720, 1280, 406, 720, False, None, None, None, 0.00486, 0.995373, 22.0, 0.21964, 'h264', 'Constrained Baseline', 24.0, 19.833, False, 1.0),
)

RECIPE_CATALOG = {}
for values in RECIPE_TUPLES:
    recipe = dict(zip(RECIPE_FIELDS, values))
    key = (recipe['platform'], recipe['modality'], recipe['quality'])
    cell = f"{recipe['band']}/{recipe['orientation']}"
    RECIPE_CATALOG.setdefault(key, {}).setdefault(cell, []).append(recipe)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def inum(v, default=None):
    if v is None or v == '':
        return default
    try:
        return int(round(float(v)))
    except Exception:
        return default


def fnum(v, default=None):
    if v is None or v == '':
        return default
    try:
        return float(v)
    except Exception:
        return default


def flag(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {'1','true','t','yes','y'}


def nearest_even(x: float) -> int:
    n = max(2, int(round(x)))
    return n if n % 2 == 0 else n + 1


def stable_seed(base_seed: int, epoch: int, key: str, copy_index: int = 0) -> int:
    digest = hashlib.sha256(f'{base_seed}|{epoch}|{key}|{copy_index}'.encode('utf8')).digest()
    return int.from_bytes(digest[:8], 'big')


def choose_platform(rng: random.Random, mixture: Optional[dict] = None, force: Optional[str] = None) -> str:
    if force is not None:
        force = force.lower()
        if force not in PLATFORM_MIXTURE_DEFAULT:
            raise ValueError(f'unsupported platform {force}')
        return force
    mixture = dict(PLATFORM_MIXTURE_DEFAULT if mixture is None else mixture)
    keys = ['clean','facebook','instagram','kakao']
    return rng.choices(keys, [mixture[k] for k in keys], k=1)[0]


def choose_quality(platform: str, modality: str, rng: random.Random, force: Optional[str] = None,
                   quality_weights: Optional[dict] = None) -> str:
    if platform == 'clean':
        return 'clean'
    if force is not None:
        return force.lower()
    weights = dict(QUALITY_WEIGHTS_DEFAULT if quality_weights is None else quality_weights)
    key = (platform, modality)
    opts = weights[key]
    qs = list(opts.keys())
    ws = list(opts.values())
    return rng.choices(qs, ws, k=1)[0]


def infer_orientation(width: int, height: int) -> str:
    return 'land' if width > height else 'port' if height > width else 'square'


def infer_band(modality: str, width: int, height: int) -> str:
    if modality == 'image':
        long_edge = max(width, height)
        if long_edge <= 768:
            return 'le512'
        if long_edge <= 1536:
            return 'le1080'
        return 'le2048'
    short_edge = min(width, height)
    if short_edge <= 900:
        return '720p'
    if short_edge <= 1536:
        return '1080p'
    return '2160p'


def matching_cells(platform: str, modality: str, quality: str, band: str, orientation: str):
    key = (platform, modality, quality)
    cells = RECIPE_CATALOG.get(key)
    if not cells:
        raise ValueError(f'No catalog for {key}')
    primary = f'{band}/{orientation}'
    if primary in cells:
        return cells[primary]
    raise ValueError(f'No recipe cell for {(platform, modality, quality, band, orientation)}')


def choose_recipe(platform: str, modality: str, quality: str, width: int, height: int,
                  rng: random.Random):
    band = infer_band(modality, width, height)
    orient = infer_orientation(width, height)
    recipes = matching_cells(platform, modality, quality, band, orient)
    # Match measured aspect ratio conditionally, not merely orientation.
    aspect = math.log(width / height)
    distances = [abs(aspect - math.log(r['orig_width']/r['orig_height'])) for r in recipes]
    closest = min(distances)
    if closest > 0.15:
        raise ValueError(f'No observed aspect ratio near {width}x{height} in {platform}/{modality}/{quality}/{band}/{orient}')
    candidates = [r for r, dist in zip(recipes, distances) if dist <= closest + 0.025]
    picked = rng.choices(candidates, [float(r.get('weight', 1.0)) for r in candidates], k=1)[0]
    return dict(picked), band, orient


def sample_policy(modality: str, width: int, height: int, *, key: str,
                  base_seed: int = 42, epoch: int = 0, copy_index: int = 0,
                  force_platform: Optional[str] = None, force_quality: Optional[str] = None,
                  mixture: Optional[dict] = None, quality_weights: Optional[dict] = None):
    rng = random.Random(stable_seed(base_seed, epoch, key, copy_index))
    platform = choose_platform(rng, mixture=mixture, force=force_platform)
    if platform == 'clean':
        return {
            'platform': 'clean', 'quality': 'clean', 'band': infer_band(modality, width, height),
            'orientation': infer_orientation(width, height), 'recipe': None, 'rng': rng,
        }
    quality = choose_quality(platform, modality, rng, force=force_quality, quality_weights=quality_weights)
    recipe, band, orient = choose_recipe(platform, modality, quality, width, height, rng)
    return {'platform': platform, 'quality': quality, 'band': band, 'orientation': orient,
            'recipe': recipe, 'rng': rng}


def list_profile_keys():
    return sorted(RECIPE_CATALOG.keys())


def count_recipes():
    return sum(len(recs) for cells in RECIPE_CATALOG.values() for recs in cells.values())


def summarize():
    lines = []
    total = 0
    for key in list_profile_keys():
        cells = RECIPE_CATALOG[key]
        n = sum(len(recs) for recs in cells.values())
        total += n
        lines.append(f'{key}: {len(cells)} cells, {n} recipes')
    lines.append(f'Total recipes: {total}')
    return '\n'.join(lines)