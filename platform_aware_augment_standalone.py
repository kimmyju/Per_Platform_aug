#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline platform-aware augmentation with EMBEDDED empirical policies.

No CSV is read at execution time. Frozen statistics come from 12 source CSV
profiles (360/480 measured pairs; Instagram repeated qualities collapsed).
This is an aggregated APPROXIMATION; it does not claim pixel/byte-level platform
replication. Measurement counts are conditional on input band/orientation.

python platform_aware_augment_standalone.py --show-profiles
python platform_aware_augment_standalone.py --input train/ --output out/ --copies 3
python platform_aware_augment_standalone.py --input train/ --output out/ --force-platform facebook --force-quality low

Default jitter=0; proposed sampling ratios Clean/FB/IG/KT=25% each are NOT
measurement-derived and should be ablated. Photometric pre-encoding correction
is opt-in because it may double-count codec-induced brightness and contrast.
For reproducible forensic evaluation, compare generated outputs against held-out
measurement distributions offline; no measurement CSV is needed for training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


# READ-ONLY EMBEDDED POLICY, automatically aggregated from measured CSVs.
# Contains only group counts and median values: no stimulus IDs or raw samples.
# CSV provenance: SHA-256 below. Build artifact is stable for repeatable training.
EMBEDDED_SOURCE_SHA256 = {'facebook_high_image_measurement.csv': '5bb151ab4602933434d02ec461efa2278d09459708d6d9e5e37fcb245136dde9',
 'facebook_high_video_measurement.csv': 'b75e622bbe73aa3126eabd594c3b76856f5baa3cfc319abe0a22a1ee00452a8e',
 'facebook_low_image_measurement.csv': '0fddd9ea8b147c1e6dbb5242f20f9655132edbc536f3533300a81f9b05cabb55',
 'facebook_low_video_measurement.csv': 'f5b6803b38704f06057f237b3a93ecc4ca1f81135308ae9498ccc768b30ab239',
 'facebook_mid_image_measurement.csv': '43c83c30e8cf42fdc102a7536eb7dd355a22496a966b756bea727aed19eba283',
 'facebook_mid_video_measurement.csv': '304d8acac96b9c3cdcc78df55b0c598ea9d58cd01f8602c04185e2c1ccdbdd03',
 'instagram_high_image_measurement.csv': 'aa0a35bcd7b883e47556069e4370fc564e65c3dccbcb19013f4a048c9d1bda93',
 'instagram_high_video_measurement.csv': '101624e7ee01201cbde78bec270359df7df9420489a442cd32d08c81ca880d7b',
 'kakao_low_image_measurement.csv': '8a99a9ae954821174d36a557fce4476f0b3baa1442e780cf67a3ac5e2654f319',
 'kakao_normal_image_measurement.csv': '6e3e595a2f6acb15f38e60c3d9cc194154089f16e09b0307fd48b40b347ebf53',
 'kakao_original_image_measurement.csv': '08b6287f25cb4d1b8bdad31721ab92f71fc697831099a0e786baf345e389d2a2',
 'kakao_video_measurement.csv': '953a36be0a7e4f7d16bd7e8f29005c2c0e3be03bd509f304ec938138f8e2a386'}
EMBEDDED_POLICY = {'facebook|image|high': {'le1080/land': [{'a': 1.0,
                                          'b': 0.0,
                                          'count': 7,
                                          'crop': 'False',
                                          'down_h': 608,
                                          'down_w': 1080,
                                          'has_icc': 'True',
                                          'progressive': 'False',
                                          'qf': 98.0,
                                          'subsamp': '4:4:4',
                                          'sx': 1.0,
                                          'sy': 1.0,
                                          'verdict': 'container_or_metadata_only'}],
                         'le1080/port': [{'a': 1.0,
                                          'b': 0.0,
                                          'count': 3,
                                          'crop': 'False',
                                          'down_h': 1080,
                                          'down_w': 767,
                                          'has_icc': 'True',
                                          'progressive': 'False',
                                          'qf': 98.0,
                                          'subsamp': '4:4:4',
                                          'sx': 1.0,
                                          'sy': 1.0,
                                          'verdict': 'container_or_metadata_only'}],
                         'le2048/land': [{'a': 1.000247,
                                          'b': -0.000895,
                                          'count': 4,
                                          'crop': 'False',
                                          'down_h': 960,
                                          'down_w': 1440,
                                          'has_icc': 'False',
                                          'progressive': 'True',
                                          'qf': 93.0,
                                          'subsamp': '4:2:0',
                                          'sx': 0.703125,
                                          'sy': 0.7032967,
                                          'verdict': 'resized_and_recompressed'}],
                         'le2048/port': [{'a': 1.0,
                                          'b': 0.0,
                                          'count': 6,
                                          'crop': 'False',
                                          'down_h': 2048,
                                          'down_w': 1365,
                                          'has_icc': 'True',
                                          'progressive': 'False',
                                          'qf': 98.0,
                                          'subsamp': '4:4:4',
                                          'sx': 1.0,
                                          'sy': 1.0,
                                          'verdict': 'container_or_metadata_only'}],
                         'le512/port': [{'a': 1.0,
                                         'b': 0.0,
                                         'count': 10,
                                         'crop': 'False',
                                         'down_h': 512,
                                         'down_w': 443,
                                         'has_icc': 'True',
                                         'progressive': 'False',
                                         'qf': 98.0,
                                         'subsamp': '4:4:4',
                                         'sx': 1.0,
                                         'sy': 1.0,
                                         'verdict': 'container_or_metadata_only'}]},
 'facebook|image|low': {'le1080/land': [{'a': 1.000328,
                                         'b': -0.001578,
                                         'count': 7,
                                         'crop': 'False',
                                         'down_h': 360,
                                         'down_w': 640,
                                         'has_icc': 'False',
                                         'progressive': 'True',
                                         'qf': 93.0,
                                         'subsamp': '4:2:0',
                                         'sx': 0.5925926,
                                         'sy': 0.5921053,
                                         'verdict': 'resized_and_recompressed'}],
                        'le1080/port': [{'a': 1.000126,
                                         'b': -0.001523,
                                         'count': 3,
                                         'crop': 'False',
                                         'down_h': 676,
                                         'down_w': 480,
                                         'has_icc': 'False',
                                         'progressive': 'True',
                                         'qf': 93.0,
                                         'subsamp': '4:2:0',
                                         'sx': 0.6258149,
                                         'sy': 0.6259259,
                                         'verdict': 'resized_and_recompressed'}],
                        'le2048/land': [{'a': 1.000646,
                                         'b': -0.001057,
                                         'count': 4,
                                         'crop': 'False',
                                         'down_h': 368,
                                         'down_w': 552,
                                         'has_icc': 'False',
                                         'progressive': 'True',
                                         'qf': 93.0,
                                         'subsamp': '4:2:0',
                                         'sx': 0.2695312,
                                         'sy': 0.2695971,
                                         'verdict': 'resized_and_recompressed'}],
                        'le2048/port': [{'a': 1.000142,
                                         'b': -0.000421,
                                         'count': 6,
                                         'crop': 'False',
                                         'down_h': 720,
                                         'down_w': 480,
                                         'has_icc': 'False',
                                         'progressive': 'True',
                                         'qf': 93.0,
                                         'subsamp': '4:2:0',
                                         'sx': 0.3516484,
                                         'sy': 0.3515625,
                                         'verdict': 'resized_and_recompressed'}],
                        'le512/port': [{'a': 1.0,
                                        'b': 0.0,
                                        'count': 10,
                                        'crop': 'False',
                                        'down_h': 512,
                                        'down_w': 443,
                                        'has_icc': 'True',
                                        'progressive': 'False',
                                        'qf': 98.0,
                                        'subsamp': '4:4:4',
                                        'sx': 1.0,
                                        'sy': 1.0,
                                        'verdict': 'container_or_metadata_only'}]},
 'facebook|image|mid': {'le1080/land': [{'a': 1.0,
                                         'b': 0.0,
                                         'count': 1,
                                         'crop': 'False',
                                         'down_h': 594,
                                         'down_w': 1080,
                                         'has_icc': 'True',
                                         'progressive': 'False',
                                         'qf': 98.0,
                                         'subsamp': '4:4:4',
                                         'sx': 1.0,
                                         'sy': 1.0,
                                         'verdict': 'container_or_metadata_only'},
                                        {'a': 0.999278,
                                         'b': 7.8e-05,
                                         'count': 6,
                                         'crop': 'False',
                                         'down_h': 600,
                                         'down_w': 1066,
                                         'has_icc': 'False',
                                         'progressive': 'True',
                                         'qf': 93.0,
                                         'subsamp': '4:2:0',
                                         'sx': 0.987037,
                                         'sy': 0.9868421,
                                         'verdict': 'resized_and_recompressed'}],
                        'le1080/port': [{'a': 1.0,
                                         'b': 0.0,
                                         'count': 2,
                                         'crop': 'False',
                                         'down_h': 1080,
                                         'down_w': 744,
                                         'has_icc': 'True',
                                         'progressive': 'False',
                                         'qf': 98.0,
                                         'subsamp': '4:4:4',
                                         'sx': 1.0,
                                         'sy': 1.0,
                                         'verdict': 'container_or_metadata_only'},
                                        {'a': 1.00043,
                                         'b': -7e-05,
                                         'count': 1,
                                         'crop': 'False',
                                         'down_h': 1079,
                                         'down_w': 843,
                                         'has_icc': 'False',
                                         'progressive': 'True',
                                         'qf': 93.0,
                                         'subsamp': '4:2:0',
                                         'sx': 0.9988152,
                                         'sy': 0.9990741,
                                         'verdict': 'resized_and_recompressed'}],
                        'le2048/land': [{'a': 1.000575,
                                         'b': -0.00162,
                                         'count': 4,
                                         'crop': 'False',
                                         'down_h': 600,
                                         'down_w': 900,
                                         'has_icc': 'False',
                                         'progressive': 'True',
                                         'qf': 93.0,
                                         'subsamp': '4:2:0',
                                         'sx': 0.4394531,
                                         'sy': 0.4395604,
                                         'verdict': 'resized_and_recompressed'}],
                        'le2048/port': [{'a': 1.000215,
                                         'b': -0.001283,
                                         'count': 6,
                                         'crop': 'False',
                                         'down_h': 1265,
                                         'down_w': 843,
                                         'has_icc': 'False',
                                         'progressive': 'True',
                                         'qf': 93.0,
                                         'subsamp': '4:2:0',
                                         'sx': 0.6175824,
                                         'sy': 0.6176758,
                                         'verdict': 'resized_and_recompressed'}],
                        'le512/port': [{'a': 1.0,
                                        'b': 0.0,
                                        'count': 10,
                                        'crop': 'False',
                                        'down_h': 512,
                                        'down_w': 443,
                                        'has_icc': 'True',
                                        'progressive': 'False',
                                        'qf': 98.0,
                                        'subsamp': '4:4:4',
                                        'sx': 1.0,
                                        'sy': 1.0,
                                        'verdict': 'container_or_metadata_only'}]},
 'facebook|video|high': {'1080p/land': [{'a': 0.998476,
                                         'b': 0.004176,
                                         'bframes': 'True',
                                         'bitrate_ratio': 0.08606,
                                         'codec': 'h264',
                                         'count': 7,
                                         'down_h': 720,
                                         'down_w': 1280,
                                         'effective_crf': 25.63,
                                         'fps': 30.0,
                                         'gop': 150.0,
                                         'sx': 0.6666667,
                                         'sy': 0.6666667,
                                         'temporal_offset': 0.0,
                                         'verdict': 'resized_and_recompressed',
                                         'vprofile': 'High'}],
                         '1080p/port': [{'a': 0.999283,
                                         'b': 0.00342,
                                         'bframes': 'True',
                                         'bitrate_ratio': 0.18336,
                                         'codec': 'h264',
                                         'count': 3,
                                         'down_h': 1280,
                                         'down_w': 720,
                                         'effective_crf': 26.0,
                                         'fps': 30.0,
                                         'gop': 150.0,
                                         'sx': 0.6666667,
                                         'sy': 0.6666667,
                                         'temporal_offset': 0.0,
                                         'verdict': 'resized_and_recompressed',
                                         'vprofile': 'High'}],
                         '2160p/land': [{'a': 0.999773,
                                         'b': 0.002888,
                                         'bframes': 'True',
                                         'bitrate_ratio': 0.059515,
                                         'codec': 'h264',
                                         'count': 4,
                                         'down_h': 720,
                                         'down_w': 1280,
                                         'effective_crf': 25.995,
                                         'fps': 30.0,
                                         'gop': 150.0,
                                         'sx': 0.3333333,
                                         'sy': 0.3333333,
                                         'temporal_offset': 0.0,
                                         'verdict': 'resized_and_recompressed',
                                         'vprofile': 'High'}],
                         '2160p/port': [{'a': 0.998908,
                                         'b': 0.00453,
                                         'bframes': 'True',
                                         'bitrate_ratio': 0.05792,
                                         'codec': 'h264',
                                         'count': 6,
                                         'down_h': 1280,
                                         'down_w': 720,
                                         'effective_crf': 25.37,
                                         'fps': 30.0,
                                         'gop': 150.0,
                                         'sx': 0.3333333,
                                         'sy': 0.3333333,
                                         'temporal_offset': 0.0,
                                         'verdict': 'resized_and_recompressed',
                                         'vprofile': 'High'}],
                         '720p/land': [{'a': 0.999625,
                                        'b': 0.000703,
                                        'bframes': 'True',
                                        'bitrate_ratio': 0.34625,
                                        'codec': 'h264',
                                        'count': 7,
                                        'down_h': 720,
                                        'down_w': 1280,
                                        'effective_crf': 25.0,
                                        'fps': 30.0,
                                        'gop': 150.0,
                                        'sx': 1.0,
                                        'sy': 1.0,
                                        'temporal_offset': 0.0,
                                        'verdict': 'recompressed_same_geometry',
                                        'vprofile': 'High'}],
                         '720p/port': [{'a': 0.999863,
                                        'b': 0.000329,
                                        'bframes': 'True',
                                        'bitrate_ratio': 0.36243,
                                        'codec': 'h264',
                                        'count': 3,
                                        'down_h': 1280,
                                        'down_w': 720,
                                        'effective_crf': 25.26,
                                        'fps': 30.0,
                                        'gop': 150.0,
                                        'sx': 1.0,
                                        'sy': 1.0,
                                        'temporal_offset': 0.0,
                                        'verdict': 'recompressed_same_geometry',
                                        'vprofile': 'High'}]},
 'facebook|video|low': {'1080p/land': [{'a': 0.996766,
                                        'b': 0.005297,
                                        'bframes': 'True',
                                        'bitrate_ratio': 0.01866,
                                        'codec': 'h264',
                                        'count': 7,
                                        'down_h': 360,
                                        'down_w': 640,
                                        'effective_crf': 28.4,
                                        'fps': 30.0,
                                        'gop': 150.0,
                                        'sx': 0.3333333,
                                        'sy': 0.3333333,
                                        'temporal_offset': 0.0,
                                        'verdict': 'resized_and_recompressed',
                                        'vprofile': 'Main'}],
                        '1080p/port': [{'a': 0.99721,
                                        'b': 0.005943,
                                        'bframes': 'True',
                                        'bitrate_ratio': 0.04204,
                                        'codec': 'h264',
                                        'count': 3,
                                        'down_h': 640,
                                        'down_w': 360,
                                        'effective_crf': 30.0,
                                        'fps': 30.0,
                                        'gop': 150.0,
                                        'sx': 0.3333333,
                                        'sy': 0.3333333,
                                        'temporal_offset': 0.0,
                                        'verdict': 'resized_and_recompressed',
                                        'vprofile': 'Main'}],
                        '2160p/land': [{'a': 0.998289,
                                        'b': 0.003724,
                                        'bframes': 'True',
                                        'bitrate_ratio': 0.01256,
                                        'codec': 'h264',
                                        'count': 4,
                                        'down_h': 360,
                                        'down_w': 640,
                                        'effective_crf': 29.355,
                                        'fps': 30.0,
                                        'gop': 150.0,
                                        'sx': 0.1666667,
                                        'sy': 0.1666667,
                                        'temporal_offset': 0.0,
                                        'verdict': 'resized_and_recompressed',
                                        'vprofile': 'Main'}],
                        '2160p/port': [{'a': 0.997046,
                                        'b': 0.006783,
                                        'bframes': 'True',
                                        'bitrate_ratio': 0.01105,
                                        'codec': 'h264',
                                        'count': 6,
                                        'down_h': 640,
                                        'down_w': 360,
                                        'effective_crf': 29.0,
                                        'fps': 30.0,
                                        'gop': 150.0,
                                        'sx': 0.1666667,
                                        'sy': 0.1666667,
                                        'temporal_offset': 0.0,
                                        'verdict': 'resized_and_recompressed',
                                        'vprofile': 'Main'}],
                        '720p/land': [{'a': 0.998193,
                                       'b': 0.004734,
                                       'bframes': 'True',
                                       'bitrate_ratio': 0.07274,
                                       'codec': 'h264',
                                       'count': 7,
                                       'down_h': 360,
                                       'down_w': 640,
                                       'effective_crf': 29.54,
                                       'fps': 30.0,
                                       'gop': 150.0,
                                       'sx': 0.5,
                                       'sy': 0.5,
                                       'temporal_offset': 0.0,
                                       'verdict': 'resized_and_recompressed',
                                       'vprofile': 'Main'}],
                        '720p/port': [{'a': 0.997414,
                                       'b': 0.004237,
                                       'bframes': 'True',
                                       'bitrate_ratio': 0.07996,
                                       'codec': 'h264',
                                       'count': 3,
                                       'down_h': 640,
                                       'down_w': 360,
                                       'effective_crf': 29.0,
                                       'fps': 30.0,
                                       'gop': 150.0,
                                       'sx': 0.5,
                                       'sy': 0.5,
                                       'temporal_offset': 0.0,
                                       'verdict': 'resized_and_recompressed',
                                       'vprofile': 'Main'}]},
 'facebook|video|mid': {'1080p/land': [{'a': 0.996766,
                                        'b': 0.005297,
                                        'bframes': 'True',
                                        'bitrate_ratio': 0.01866,
                                        'codec': 'h264',
                                        'count': 7,
                                        'down_h': 360,
                                        'down_w': 640,
                                        'effective_crf': 28.4,
                                        'fps': 30.0,
                                        'gop': 150.0,
                                        'sx': 0.3333333,
                                        'sy': 0.3333333,
                                        'temporal_offset': 0.0,
                                        'verdict': 'resized_and_recompressed',
                                        'vprofile': 'Main'}],
                        '1080p/port': [{'a': 0.99721,
                                        'b': 0.005943,
                                        'bframes': 'True',
                                        'bitrate_ratio': 0.04204,
                                        'codec': 'h264',
                                        'count': 3,
                                        'down_h': 640,
                                        'down_w': 360,
                                        'effective_crf': 30.0,
                                        'fps': 30.0,
                                        'gop': 150.0,
                                        'sx': 0.3333333,
                                        'sy': 0.3333333,
                                        'temporal_offset': 0.0,
                                        'verdict': 'resized_and_recompressed',
                                        'vprofile': 'Main'}],
                        '2160p/land': [{'a': 0.998289,
                                        'b': 0.003724,
                                        'bframes': 'True',
                                        'bitrate_ratio': 0.01256,
                                        'codec': 'h264',
                                        'count': 4,
                                        'down_h': 360,
                                        'down_w': 640,
                                        'effective_crf': 29.355,
                                        'fps': 30.0,
                                        'gop': 150.0,
                                        'sx': 0.1666667,
                                        'sy': 0.1666667,
                                        'temporal_offset': 0.0,
                                        'verdict': 'resized_and_recompressed',
                                        'vprofile': 'Main'}],
                        '2160p/port': [{'a': 0.997046,
                                        'b': 0.006783,
                                        'bframes': 'True',
                                        'bitrate_ratio': 0.01105,
                                        'codec': 'h264',
                                        'count': 6,
                                        'down_h': 640,
                                        'down_w': 360,
                                        'effective_crf': 29.0,
                                        'fps': 30.0,
                                        'gop': 150.0,
                                        'sx': 0.1666667,
                                        'sy': 0.1666667,
                                        'temporal_offset': 0.0,
                                        'verdict': 'resized_and_recompressed',
                                        'vprofile': 'Main'}],
                        '720p/land': [{'a': 0.998193,
                                       'b': 0.004734,
                                       'bframes': 'True',
                                       'bitrate_ratio': 0.07274,
                                       'codec': 'h264',
                                       'count': 7,
                                       'down_h': 360,
                                       'down_w': 640,
                                       'effective_crf': 29.54,
                                       'fps': 30.0,
                                       'gop': 150.0,
                                       'sx': 0.5,
                                       'sy': 0.5,
                                       'temporal_offset': 0.0,
                                       'verdict': 'resized_and_recompressed',
                                       'vprofile': 'Main'}],
                        '720p/port': [{'a': 0.997414,
                                       'b': 0.004237,
                                       'bframes': 'True',
                                       'bitrate_ratio': 0.07996,
                                       'codec': 'h264',
                                       'count': 3,
                                       'down_h': 640,
                                       'down_w': 360,
                                       'effective_crf': 29.0,
                                       'fps': 30.0,
                                       'gop': 150.0,
                                       'sx': 0.5,
                                       'sy': 0.5,
                                       'temporal_offset': 0.0,
                                       'verdict': 'resized_and_recompressed',
                                       'vprofile': 'Main'}]},
 'instagram|image|shared': {'le1080/land': [{'a': 1.00002,
                                             'b': -7.5e-05,
                                             'count': 7,
                                             'crop': 'True',
                                             'down_h': 608,
                                             'down_w': 608,
                                             'has_icc': 'False',
                                             'ig_crop_factor': 1.0,
                                             'progressive': 'True',
                                             'qf': 92.0,
                                             'subsamp': '4:2:0',
                                             'sx': 0.562963,
                                             'sy': 1.0,
                                             'verdict': 'resized_and_recompressed'}],
                            'le1080/port': [{'a': 1.000108,
                                             'b': -4.3e-05,
                                             'count': 3,
                                             'crop': 'True',
                                             'down_h': 767,
                                             'down_w': 767,
                                             'has_icc': 'False',
                                             'ig_crop_factor': 1.0,
                                             'progressive': 'True',
                                             'qf': 92.0,
                                             'subsamp': '4:2:0',
                                             'sx': 1.0,
                                             'sy': 0.7101852,
                                             'verdict': 'resized_and_recompressed'}],
                            'le2048/land': [{'a': 1.000088,
                                             'b': -0.001353,
                                             'count': 4,
                                             'crop': 'True',
                                             'down_h': 1365,
                                             'down_w': 1365,
                                             'has_icc': 'False',
                                             'ig_crop_factor': 1.0,
                                             'progressive': 'True',
                                             'qf': 92.0,
                                             'subsamp': '4:2:0',
                                             'sx': 0.6665039,
                                             'sy': 1.0,
                                             'verdict': 'resized_and_recompressed'}],
                            'le2048/port': [{'a': 0.999905,
                                             'b': -0.002733,
                                             'count': 6,
                                             'crop': 'True',
                                             'down_h': 1365,
                                             'down_w': 1365,
                                             'has_icc': 'False',
                                             'ig_crop_factor': 1.0,
                                             'progressive': 'True',
                                             'qf': 92.0,
                                             'subsamp': '4:2:0',
                                             'sx': 1.0,
                                             'sy': 0.6665039,
                                             'verdict': 'resized_and_recompressed'}],
                            'le512/port': [{'a': 0.999972,
                                            'b': -0.002465,
                                            'count': 10,
                                            'crop': 'True',
                                            'down_h': 443,
                                            'down_w': 443,
                                            'has_icc': 'False',
                                            'ig_crop_factor': 1.0,
                                            'progressive': 'True',
                                            'qf': 92.0,
                                            'subsamp': '4:2:0',
                                            'sx': 1.0,
                                            'sy': 0.8652344,
                                            'verdict': 'resized_and_recompressed'}]},
 'instagram|video|shared': {'1080p/land': [{'a': 0.998368,
                                            'b': 0.005145,
                                            'bframes': 'True',
                                            'bitrate_ratio': 0.06523,
                                            'codec': 'h264',
                                            'count': 7,
                                            'down_h': 720,
                                            'down_w': 720,
                                            'effective_crf': 25.73,
                                            'fps': 30.0,
                                            'gop': 150.0,
                                            'sx': 0.375,
                                            'sy': 0.6666667,
                                            'temporal_offset': 0.0,
                                            'verdict': 'resized_and_recompressed',
                                            'vprofile': 'High'}],
                            '1080p/port': [{'a': 0.999256,
                                            'b': 0.003807,
                                            'bframes': 'True',
                                            'bitrate_ratio': 0.11374,
                                            'codec': 'h264',
                                            'count': 3,
                                            'down_h': 720,
                                            'down_w': 720,
                                            'effective_crf': 26.25,
                                            'fps': 30.0,
                                            'gop': 150.0,
                                            'sx': 0.6666667,
                                            'sy': 0.375,
                                            'temporal_offset': 0.0,
                                            'verdict': 'resized_and_recompressed',
                                            'vprofile': 'High'}],
                            '2160p/land': [{'a': 0.999495,
                                            'b': 0.002945,
                                            'bframes': 'True',
                                            'bitrate_ratio': 0.033365,
                                            'codec': 'h264',
                                            'count': 4,
                                            'down_h': 720,
                                            'down_w': 720,
                                            'effective_crf': 25.83,
                                            'fps': 30.0,
                                            'gop': 150.0,
                                            'sx': 0.1875,
                                            'sy': 0.3333333,
                                            'temporal_offset': 0.0,
                                            'verdict': 'resized_and_recompressed',
                                            'vprofile': 'High'}],
                            '2160p/port': [{'a': 0.998835,
                                            'b': 0.004747,
                                            'bframes': 'True',
                                            'bitrate_ratio': 0.03425,
                                            'codec': 'h264',
                                            'count': 6,
                                            'down_h': 720,
                                            'down_w': 720,
                                            'effective_crf': 25.58,
                                            'fps': 30.0,
                                            'gop': 150.0,
                                            'sx': 0.3333333,
                                            'sy': 0.1875,
                                            'temporal_offset': 0.0,
                                            'verdict': 'resized_and_recompressed',
                                            'vprofile': 'High'}],
                            '720p/land': [{'a': 0.999281,
                                           'b': 0.001125,
                                           'bframes': 'True',
                                           'bitrate_ratio': 0.2474,
                                           'codec': 'h264',
                                           'count': 7,
                                           'down_h': 720,
                                           'down_w': 720,
                                           'effective_crf': 24.7,
                                           'fps': 30.0,
                                           'gop': 150.0,
                                           'sx': 0.5625,
                                           'sy': 1.0,
                                           'temporal_offset': 0.0,
                                           'verdict': 'resized_and_recompressed',
                                           'vprofile': 'High'}],
                            '720p/port': [{'a': 0.999798,
                                           'b': 0.000618,
                                           'bframes': 'True',
                                           'bitrate_ratio': 0.22879,
                                           'codec': 'h264',
                                           'count': 3,
                                           'down_h': 720,
                                           'down_w': 720,
                                           'effective_crf': 25.51,
                                           'fps': 30.0,
                                           'gop': 150.0,
                                           'sx': 1.0,
                                           'sy': 0.5625,
                                           'temporal_offset': 0.0,
                                           'verdict': 'resized_and_recompressed',
                                           'vprofile': 'High'}]},
 'kakao|image|low': {'le1080/land': [{'a': 1.000677,
                                      'b': -0.002912,
                                      'count': 7,
                                      'crop': 'False',
                                      'down_h': 541,
                                      'down_w': 960,
                                      'has_icc': 'False',
                                      'progressive': 'False',
                                      'qf': 86.0,
                                      'subsamp': '4:2:0',
                                      'sx': 0.8888889,
                                      'sy': 0.8898026,
                                      'verdict': 'resized_and_recompressed'}],
                     'le1080/port': [{'a': 1.001218,
                                      'b': -0.002514,
                                      'count': 3,
                                      'crop': 'False',
                                      'down_h': 960,
                                      'down_w': 682,
                                      'has_icc': 'False',
                                      'progressive': 'False',
                                      'qf': 86.0,
                                      'subsamp': '4:2:0',
                                      'sx': 0.8891786,
                                      'sy': 0.8888889,
                                      'verdict': 'resized_and_recompressed'}],
                     'le2048/land': [{'a': 1.000399,
                                      'b': -0.005307,
                                      'count': 4,
                                      'crop': 'False',
                                      'down_h': 640,
                                      'down_w': 960,
                                      'has_icc': 'False',
                                      'progressive': 'False',
                                      'qf': 86.0,
                                      'subsamp': '4:2:0',
                                      'sx': 0.46875,
                                      'sy': 0.4688645,
                                      'verdict': 'resized_and_recompressed'}],
                     'le2048/port': [{'a': 1.001056,
                                      'b': -0.004211,
                                      'count': 6,
                                      'crop': 'False',
                                      'down_h': 960,
                                      'down_w': 640,
                                      'has_icc': 'False',
                                      'progressive': 'False',
                                      'qf': 86.0,
                                      'subsamp': '4:2:0',
                                      'sx': 0.4688645,
                                      'sy': 0.46875,
                                      'verdict': 'resized_and_recompressed'}],
                     'le512/port': [{'a': 0.999529,
                                     'b': -0.000972,
                                     'count': 10,
                                     'crop': 'False',
                                     'down_h': 512,
                                     'down_w': 443,
                                     'has_icc': 'True',
                                     'progressive': 'False',
                                     'qf': 86.0,
                                     'subsamp': '4:2:0',
                                     'sx': 1.0,
                                     'sy': 1.0,
                                     'verdict': 'recompressed_same_geometry'}]},
 'kakao|image|normal': {'le1080/land': [{'a': 1.000332,
                                         'b': -0.001885,
                                         'count': 7,
                                         'crop': 'False',
                                         'down_h': 608,
                                         'down_w': 1080,
                                         'has_icc': 'True',
                                         'progressive': 'False',
                                         'qf': 96.0,
                                         'subsamp': '4:2:0',
                                         'sx': 1.0,
                                         'sy': 1.0,
                                         'verdict': 'recompressed_same_geometry'}],
                        'le1080/port': [{'a': 1.000278,
                                         'b': -0.001866,
                                         'count': 3,
                                         'crop': 'False',
                                         'down_h': 1080,
                                         'down_w': 767,
                                         'has_icc': 'True',
                                         'progressive': 'False',
                                         'qf': 96.0,
                                         'subsamp': '4:2:0',
                                         'sx': 1.0,
                                         'sy': 1.0,
                                         'verdict': 'recompressed_same_geometry'}],
                        'le2048/land': [{'a': 1.000255,
                                         'b': -0.00347,
                                         'count': 4,
                                         'crop': 'False',
                                         'down_h': 960,
                                         'down_w': 1440,
                                         'has_icc': 'False',
                                         'progressive': 'False',
                                         'qf': 96.0,
                                         'subsamp': '4:2:0',
                                         'sx': 0.703125,
                                         'sy': 0.7032967,
                                         'verdict': 'resized_and_recompressed'}],
                        'le2048/port': [{'a': 1.00066,
                                         'b': -0.003005,
                                         'count': 6,
                                         'crop': 'False',
                                         'down_h': 1440,
                                         'down_w': 960,
                                         'has_icc': 'False',
                                         'progressive': 'False',
                                         'qf': 96.0,
                                         'subsamp': '4:2:0',
                                         'sx': 0.7032109,
                                         'sy': 0.703125,
                                         'verdict': 'resized_and_recompressed'}],
                        'le512/port': [{'a': 0.999769,
                                        'b': -0.001104,
                                        'count': 10,
                                        'crop': 'False',
                                        'down_h': 512,
                                        'down_w': 443,
                                        'has_icc': 'True',
                                        'progressive': 'False',
                                        'qf': 96.0,
                                        'subsamp': '4:2:0',
                                        'sx': 1.0,
                                        'sy': 1.0,
                                        'verdict': 'recompressed_same_geometry'}]},
 'kakao|image|original': {'le1080/land': [{'a': 1.0,
                                           'b': 0.0,
                                           'count': 7,
                                           'crop': 'False',
                                           'down_h': 608,
                                           'down_w': 1080,
                                           'has_icc': 'True',
                                           'progressive': 'False',
                                           'qf': 98.0,
                                           'subsamp': '4:4:4',
                                           'sx': 1.0,
                                           'sy': 1.0,
                                           'verdict': 'container_or_metadata_only'}],
                          'le1080/port': [{'a': 1.0,
                                           'b': 0.0,
                                           'count': 3,
                                           'crop': 'False',
                                           'down_h': 1080,
                                           'down_w': 767,
                                           'has_icc': 'True',
                                           'progressive': 'False',
                                           'qf': 98.0,
                                           'subsamp': '4:4:4',
                                           'sx': 1.0,
                                           'sy': 1.0,
                                           'verdict': 'container_or_metadata_only'}],
                          'le2048/land': [{'a': 1.0,
                                           'b': 0.0,
                                           'count': 4,
                                           'crop': 'False',
                                           'down_h': 1365,
                                           'down_w': 2048,
                                           'has_icc': 'True',
                                           'progressive': 'False',
                                           'qf': 98.0,
                                           'subsamp': '4:4:4',
                                           'sx': 1.0,
                                           'sy': 1.0,
                                           'verdict': 'container_or_metadata_only'}],
                          'le2048/port': [{'a': 1.0,
                                           'b': 0.0,
                                           'count': 6,
                                           'crop': 'False',
                                           'down_h': 2048,
                                           'down_w': 1365,
                                           'has_icc': 'True',
                                           'progressive': 'False',
                                           'qf': 98.0,
                                           'subsamp': '4:4:4',
                                           'sx': 1.0,
                                           'sy': 1.0,
                                           'verdict': 'container_or_metadata_only'}],
                          'le512/port': [{'a': 1.0,
                                          'b': 0.0,
                                          'count': 10,
                                          'crop': 'False',
                                          'down_h': 512,
                                          'down_w': 443,
                                          'has_icc': 'True',
                                          'progressive': 'False',
                                          'qf': 98.0,
                                          'subsamp': '4:4:4',
                                          'sx': 1.0,
                                          'sy': 1.0,
                                          'verdict': 'container_or_metadata_only'}]},
 'kakao|video|single': {'1080p/land': [{'a': 0.999786,
                                        'b': 0.000164,
                                        'bframes': 'True',
                                        'bitrate_ratio': 0.36636,
                                        'codec': 'hevc',
                                        'count': 7,
                                        'down_h': 1080,
                                        'down_w': 1920,
                                        'effective_crf': 21.0,
                                        'fps': 30.0,
                                        'gop': 28.0,
                                        'sx': 1.0,
                                        'sy': 1.0,
                                        'temporal_offset': 1.0,
                                        'verdict': 'transcoded_different_format',
                                        'vprofile': 'Main'}],
                        '1080p/port': [{'a': 0.998577,
                                        'b': 0.000388,
                                        'bframes': 'True',
                                        'bitrate_ratio': 0.25554,
                                        'codec': 'hevc',
                                        'count': 3,
                                        'down_h': 1920,
                                        'down_w': 1080,
                                        'effective_crf': 29.0,
                                        'fps': 30.0,
                                        'gop': 28.0,
                                        'sx': 1.0,
                                        'sy': 1.0,
                                        'temporal_offset': 1.0,
                                        'verdict': 'transcoded_different_format',
                                        'vprofile': 'Main'}],
                        '2160p/land': [{'a': 1.000144,
                                        'b': 0.002041,
                                        'bframes': 'True',
                                        'bitrate_ratio': 0.1284,
                                        'codec': 'hevc',
                                        'count': 4,
                                        'down_h': 1080,
                                        'down_w': 1920,
                                        'effective_crf': 24.165,
                                        'fps': 30.0,
                                        'gop': 28.0,
                                        'sx': 0.5,
                                        'sy': 0.5,
                                        'temporal_offset': 1.0,
                                        'verdict': 'transcoded_different_format',
                                        'vprofile': 'Main'}],
                        '2160p/port': [{'a': 0.998908,
                                        'b': 0.003177,
                                        'bframes': 'True',
                                        'bitrate_ratio': 0.09837,
                                        'codec': 'hevc',
                                        'count': 6,
                                        'down_h': 1920,
                                        'down_w': 1080,
                                        'effective_crf': 25.915,
                                        'fps': 30.0,
                                        'gop': 28.0,
                                        'sx': 0.5,
                                        'sy': 0.5,
                                        'temporal_offset': 1.0,
                                        'verdict': 'transcoded_different_format',
                                        'vprofile': 'Main'}],
                        '720p/land': [{'a': 0.989268,
                                       'b': 0.000504,
                                       'bframes': 'False',
                                       'bitrate_ratio': 0.35718,
                                       'codec': 'h264',
                                       'count': 7,
                                       'down_h': 720,
                                       'down_w': 1280,
                                       'effective_crf': 24.0,
                                       'fps': 24.0,
                                       'gop': 24.0,
                                       'sx': 1.0,
                                       'sy': 1.0,
                                       'temporal_offset': 4.0,
                                       'verdict': 'recompressed_same_geometry',
                                       'vprofile': 'Constrained Baseline'}],
                        '720p/port': [{'a': 0.995373,
                                       'b': 0.00486,
                                       'bframes': 'False',
                                       'bitrate_ratio': 0.28631,
                                       'codec': 'h264',
                                       'count': 3,
                                       'down_h': 720,
                                       'down_w': 406,
                                       'effective_crf': 19.36,
                                       'fps': 24.0,
                                       'gop': 20.0,
                                       'sx': 0.5638889,
                                       'sy': 0.5625,
                                       'temporal_offset': 4.0,
                                       'verdict': 'resized_and_recompressed',
                                       'vprofile': 'Constrained Baseline'}]}}

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
VIDEO_SUFFIXES = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v'}
PLATFORMS = ('clean', 'facebook', 'instagram', 'kakao')
IG_DUPLICATE_CHECK_FILES = {
    ('image', 'mid'): 'instagram_mid_image_measurement.csv',
    ('image', 'low'): 'instagram_low_image_measurement.csv',
    ('video', 'mid'): 'instagram_mid_video_measurement.csv',
    ('video', 'low'): 'instagram_low_video_measurement.csv',
}

QUALITY_FILES = {
    ('facebook', 'image', 'high'): 'facebook_high_image_measurement.csv',
    ('facebook', 'image', 'mid'): 'facebook_mid_image_measurement.csv',
    ('facebook', 'image', 'low'): 'facebook_low_image_measurement.csv',
    ('facebook', 'video', 'high'): 'facebook_high_video_measurement.csv',
    ('facebook', 'video', 'mid'): 'facebook_mid_video_measurement.csv',
    ('facebook', 'video', 'low'): 'facebook_low_video_measurement.csv',
    # IMPORTANT: only one Instagram profile for each modality to avoid
    # triple-counting high/mid/low repeated measurement outcomes.
    ('instagram', 'image', 'shared'): 'instagram_high_image_measurement.csv',
    ('instagram', 'video', 'shared'): 'instagram_high_video_measurement.csv',
    ('kakao', 'image', 'original'): 'kakao_original_image_measurement.csv',
    ('kakao', 'image', 'normal'): 'kakao_normal_image_measurement.csv',
    ('kakao', 'image', 'low'): 'kakao_low_image_measurement.csv',
    ('kakao', 'video', 'single'): 'kakao_video_measurement.csv',
}


def fnum(s: Any, default: Optional[float] = None) -> Optional[float]:
    if s is None or str(s).strip().lower() in ('', 'nan', 'none', 'n/a'):
        return default
    try:
        ans = float(s)
        return ans if math.isfinite(ans) else default
    except (ValueError, TypeError):
        return default


def inum(s: Any, default: int = 0) -> int:
    return int(round(fnum(s, default) or 0))


def flag(s: Any, default: bool = False) -> bool:
    return default if s is None or str(s) == '' else str(s).lower() in ('true', '1', 'yes')


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def nearest_even(n: float) -> int:
    return max(2, 2 * int(round(float(n) / 2)))


def stable_seed(seed: int, rel: Path, copy_index: int, epoch: int) -> int:
    digest = hashlib.sha256(
        f'{seed}|{rel.as_posix()}|{copy_index}|{epoch}'.encode('utf8')
    ).digest()
    return int.from_bytes(digest[:8], 'big')


def band_from_dimensions(modality: str, w: int, h: int) -> str:
    # Image filenames in study: ...-le512/le1080/le2048.
    # Video filenames: ...-720p/1080p/2160p.
    if modality == 'image':
        size = max(w, h)
        return 'le512' if size <= 768 else ('le1080' if size <= 1536 else 'le2048')
    # Videos include landscape 1920x1080 and portrait 1080x1920,
    # so classification must use the SHORT edge, not the long edge.
    short = min(w, h)
    return '720p' if short <= 900 else ('1080p' if short <= 1536 else '2160p')


def row_band(r: dict[str, str]) -> str:
    return r['stimulus_id'].rsplit('-', 1)[-1]


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)


def ffprobe_json(path: Path) -> dict[str, Any]:
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height,avg_frame_rate,bit_rate,nb_frames,codec_name,profile',
           '-show_entries', 'format=bit_rate,duration', '-of', 'json', str(path)]
    obj = json.loads(run(cmd).stdout)
    streams = obj.get('streams', [])
    if not streams:
        raise RuntimeError(f'No video stream: {path}')
    st = streams[0]
    rate = st.get('avg_frame_rate', '0/1')
    if '/' in str(rate):
        n, d = str(rate).split('/', 1)
        fps = float(n) / float(d) if float(d) else 0
    else:
        fps = float(rate)
    # A stream bit_rate may be missing: format bitrate can include audio,
    # and is used as a documented fallback approximation.
    in_br = fnum(st.get('bit_rate'), fnum(obj.get('format', {}).get('bit_rate'), 0))
    return {
        'width': int(st['width']), 'height': int(st['height']),
        'fps': fps, 'bitrate_bps': in_br,
        'codec': st.get('codec_name'), 'profile': st.get('profile'),
        'frames': st.get('nb_frames'),
    }


class MeasurementProfiles:
    """Frozen empirical summary; zero CSV/file IO at training runtime.

    12 groups (360 observations, Instagram high/mid/low collapsed to one
    profile per modality); counts are measured within each orientation and
    input band. The embedded numbers are MEDIANS of measured per-pair rows,
    not exact encoder parameters or proof of app policies.
    """
    def __init__(self):
        self.data = EMBEDDED_POLICY

    def show(self):
        print('PROFILE | BAND | ORIENTATION | n | VERDICT(count)')
        for name, groups in sorted(self.data.items()):
            for category, variants in sorted(groups.items()):
                counts = ', '.join(f"{v['verdict']}:{v['count']}" for v in variants)
                print(f'{name:26} {category:14} {sum(v["count"] for v in variants):2d}  {counts}')

    def choose(self, platform: str, modality: str, quality: str,
               band: str, width: int, height: int, rng: random.Random):
        key = f'{platform}|{modality}|{quality}'
        orientation = 'land' if width > height else 'port' if height > width else 'square'
        category = f'{band}/{orientation}'
        groups = self.data.get(key, {})
        if category not in groups:
            # Do not silently invent platform behavior for unobserved aspect ratios.
            raise ValueError(f'No measured profile for {key} at {category}. '
                             'Provide a supported input geometry or an explicit OOD policy.')
        variants = groups[category]
        v = rng.choices(variants, weights=[x['count'] for x in variants], k=1)[0]
        # Reconstruct the minimum schema needed by the image/video transforms.
        # Geometry uses AGGREGATE dimension ratios, not raw stimulus rows.
        row = {
            'stimulus_id': f'aggregate:{key}:{category}:{v["verdict"]}',
            'target_id': key,
            'reencode_verdict': v['verdict'],
            'orig_width': str(width), 'orig_height': str(height),
            'down_width': str(max(1, round(width * v['sx']))),
            'down_height': str(max(1, round(height * v['sy']))),
            'crop_applied': v.get('crop','False'),
            'down_estimated_jpeg_qf': str(v.get('qf','')),
            'down_chroma_subsampling': v.get('subsamp','4:2:0'),
            'down_jpeg_progressive': v.get('progressive','False'),
            'down_has_icc': v.get('has_icc','False'),
            'brightness_offset_b': str(v.get('b',0)),
            'contrast_gain_a': str(v.get('a',1)),
            'displayed_brightness_offset_b_median': str(v.get('b',0)),
            'displayed_contrast_gain_a_median': str(v.get('a',1)),
            'down_codec_name': v.get('codec','h264'),
            'down_profile': v.get('vprofile',''),
            'down_avg_frame_rate': str(v.get('fps','')),
            'down_gop_mean': str(v.get('gop',150)),
            'down_has_b_frames': v.get('bframes','False'),
            'bitrate_ratio': str(v.get('bitrate_ratio','')),
            'effective_crf_interp': str(v.get('effective_crf','')),
            'temporal_offset_frames': str(v.get('temporal_offset',0)),
        }
        if platform == 'instagram' and modality == 'image':
            # Square crop to measured fraction of input shorter edge.
            factor = v.get('ig_crop_factor',1.0)
            size = max(1,round(min(width,height)*factor))
            row['down_width'] = str(size)
            row['down_height'] = str(size)
            row['orig_width'] = str(min(width,height))
            row['orig_height'] = str(min(width,height))
        if platform == 'instagram' and modality == 'video':
            # All measured IG video outputs are exactly 720x720.
            row['down_width'] = '720'
            row['down_height'] = '720'
        return row, band


def pick_quality(platform: str, modality: str, rng: random.Random,
                 forced: Optional[str] = None) -> str:
    if platform == 'instagram':
        if forced not in (None, 'shared', 'high', 'mid', 'low'):
            raise ValueError('Instagram uses a collapsed shared profile.')
        return 'shared'
    if platform == 'kakao' and modality == 'video':
        if forced not in (None, 'single'):
            raise ValueError('KakaoTalk video only has one real measured profile.')
        return 'single'
    if platform == 'facebook' and modality == 'video':
        opts = ['high', 'mid', 'low']
        probs = [0.50, 0.25, 0.25]
    else:
        opts = ['high', 'mid', 'low'] if platform == 'facebook' else ['original', 'normal', 'low']
        probs = [1/3] * 3
    if forced is not None:
        if forced not in opts:
            raise ValueError(f'Invalid quality {forced} for {platform}/{modality}: {opts}')
        return forced
    return rng.choices(opts, probs, k=1)[0]


def photo_targets(row: dict[str, str], modality: str,
                  rng: random.Random, args: argparse.Namespace) -> dict[str, float]:
    if modality == 'image':
        b = fnum(row.get('brightness_offset_b'), 0.0) or 0.0
        a = fnum(row.get('contrast_gain_a'), 1.0) or 1.0
    else:
        b = fnum(row.get('displayed_brightness_offset_b_median'), 0.0) or 0.0
        a = fnum(row.get('displayed_contrast_gain_a_median'), 1.0) or 1.0
    if rng.random() <= args.p_photo:
        # Proposal: slight +/-10% scatter around MEASURED row values.
        jitter = args.photo_jitter
        b *= rng.uniform(1-jitter, 1+jitter)
        a = 1.0 + (a - 1.0) * rng.uniform(1-jitter, 1+jitter)
        return {'b': round(clamp(b, -0.04, 0.04), 7),
                'a': round(clamp(a, 0.95, 1.05), 7), 'applied': True}
    return {'b': 0.0, 'a': 1.0, 'applied': False}


def image_geometry(w: int, h: int, row: dict[str, str],
                   platform: str, rng: random.Random, args: argparse.Namespace
                   ) -> tuple[tuple[int, int, int, int] | None, tuple[int, int], dict]:
    # Reference dimensions represent uploaded->redownloaded mapping.
    ow = inum(row['orig_width']); oh = inum(row['orig_height'])
    dw = inum(row['down_width']); dh = inum(row['down_height'])
    crop = flag(row.get('crop_applied')) and platform == 'instagram'
    info = {'reference_size': [ow, oh], 'reference_download_size': [dw, dh],
            'measured_crop_applied': flag(row.get('crop_applied'))}
    if crop:
        square = min(w, h)
        span_x, span_y = w - square, h - square
        offx, offy = span_x / 2, span_y / 2
        jitter = rng.random() < args.ig_crop_jitter_prob
        if jitter:
            offx += rng.uniform(-1, 1) * span_x * args.ig_crop_jitter_frac
            offy += rng.uniform(-1, 1) * span_y * args.ig_crop_jitter_frac
        x = max(0, min(w - square, int(round(offx))))
        y = max(0, min(h - square, int(round(offy))))
        rect = (x, y, x + square, y + square)
        # For the observed images, output square side is generally the INPUT
        # short side; proportional adjustment accommodates other source sizes.
        short_ref = min(ow, oh)
        side_ratio = dw / max(1, short_ref)
        side = max(1, int(round(square * side_ratio)))
        info.update({'center_crop_assumed': True, 'crop_jittered': jitter,
                     'crop_box': list(rect), 'resize_to': [side, side]})
        return rect, (side, side), info
    # For an identity verdict, preserve geometry. JPEG reencode with no
    # resized output will still take this no-crop path.
    nw = max(1, int(round(w * dw / max(1, ow))))
    nh = max(1, int(round(h * dh / max(1, oh))))
    info.update({'center_crop_assumed': False, 'crop_jittered': False,
                 'crop_box': None, 'resize_to': [nw, nh]})
    return None, (nw, nh), info


def adjust_luma_pillow(img, a: float, b: float):
    """Approximate measured y=a*x+b on normalized 8-bit luma."""
    from PIL import Image
    if abs(a-1.0) < 1e-12 and abs(b) < 1e-12:
        return img
    yc = img.convert('YCbCr')
    y, cb, cr = yc.split()
    lut = [max(0, min(255, int(round(255*(a*i/255+b))))) for i in range(256)]
    return Image.merge('YCbCr', (y.point(lut), cb, cr)).convert('RGB')


def image_mask_for(image: Path, source_root: Path, mask_root: Path) -> Optional[Path]:
    rel = image.relative_to(source_root)
    m = mask_root / rel.parent / (rel.stem + '.png')
    return m if m.is_file() else None


def transform_image(src: Path, dst: Path, row: dict[str,str],
                    platform: str, rng: random.Random, args: argparse.Namespace,
                    mask_src: Optional[Path], mask_dst: Optional[Path]) -> dict:
    from PIL import Image, ImageOps
    verdict = row['reencode_verdict']
    if verdict == 'container_or_metadata_only':
        # Pixel-level identity: do not mistakenly introduce a JPEG generation.
        shutil.copy2(src, dst)
        if mask_src and mask_dst:
            mask_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mask_src, mask_dst)
        return {'verdict': verdict, 'pixel_transform': 'identity',
                'photo': {'b': 0, 'a': 1, 'applied': False}, 'jpeg': None,
                'geometry': 'preserved',
                'note': 'Metadata/container rewrite not emulated; output byte-copy.'}

    with Image.open(src) as open_img:
        im = ImageOps.exif_transpose(open_img).convert('RGB')
        orig_icc = open_img.info.get('icc_profile')
    crop_rect, target, geo = image_geometry(im.width, im.height, row, platform, rng, args)
    if crop_rect is not None:
        im = im.crop(crop_rect)
    if im.size != target:
        im = im.resize(target, Image.Resampling.LANCZOS)
    photo = photo_targets(row, 'image', rng, args)
    if photo['applied']:
        im = adjust_luma_pillow(im, photo['a'], photo['b'])
    qf = inum(row.get('down_estimated_jpeg_qf'), 95)
    qf = int(round(clamp(qf + rng.uniform(-args.jpeg_qf_jitter, args.jpeg_qf_jitter), 1, 100)))
    subsamp = row.get('down_chroma_subsampling', '4:2:0')
    subsampling = {'4:4:4': 0, '4:2:2': 1, '4:2:0': 2}.get(subsamp, 2)
    progressive = flag(row.get('down_jpeg_progressive'))
    kwargs: dict[str, Any] = {'format': 'JPEG', 'quality': qf,
                               'subsampling': subsampling, 'progressive': progressive}
    if flag(row.get('down_has_icc')) and orig_icc:
        kwargs['icc_profile'] = orig_icc
    im.save(dst, **kwargs)
    if mask_src and mask_dst:
        # Ensure masks are transformed IDENTICALLY in space, but never
        # compressed as JPEG, color-corrected, or interpolated bilinearly.
        with Image.open(mask_src) as mask_in:
            mask = mask_in.copy()
        if crop_rect is not None:
            mask = mask.crop(crop_rect)
        if mask.size != target:
            mask = mask.resize(target, Image.Resampling.NEAREST)
        mask_dst.parent.mkdir(parents=True, exist_ok=True)
        mask.save(mask_dst)
    return {'verdict': verdict, 'pixel_transform': 'resized_or_recompressed',
            'geometry': geo, 'photo': photo,
            'jpeg': {'qf': qf, 'subsampling': subsamp, 'progressive': progressive,
                     'metadata_preserved': False}}


def transform_video(src: Path, dst: Path, row: dict[str,str], platform: str,
                    rng: random.Random, args: argparse.Namespace,
                    probe: dict[str, Any]) -> dict:
    src_w, src_h = probe['width'], probe['height']
    ow = inum(row['orig_width']); oh = inum(row['orig_height'])
    dw = inum(row['down_width']); dh = inum(row['down_height'])
    filters: list[str] = []
    geometry: dict[str, Any] = {'reference_size': [ow, oh],
                               'reference_download_size': [dw, dh]}
    if platform == 'instagram':
        # IG videos 720x720 in all 30 measured pairs, per quality profile.
        # Center anchoring is an APPROXIMATION, not proven by CSV.
        min_edge = min(src_w, src_h)
        off_x = (src_w - min_edge) // 2
        off_y = (src_h - min_edge) // 2
        jitter = rng.random() < args.ig_crop_jitter_prob
        if jitter:
            off_x = round(off_x + rng.uniform(-1, 1) * (src_w-min_edge) * args.ig_crop_jitter_frac)
            off_y = round(off_y + rng.uniform(-1, 1) * (src_h-min_edge) * args.ig_crop_jitter_frac)
        off_x = max(0, min(src_w-min_edge, int(off_x)))
        off_y = max(0, min(src_h-min_edge, int(off_y)))
        filters.append(f'crop={min_edge}:{min_edge}:{off_x}:{off_y}')
        target = (nearest_even(dw), nearest_even(dh))  # measured 720x720
        geometry.update({'crop_box': [off_x, off_y, min_edge, min_edge],
                         'crop_jittered': jitter, 'center_crop_assumed': True})
    else:
        target = (nearest_even(src_w*dw/max(1,ow)),
                  nearest_even(src_h*dh/max(1,oh)))
        geometry.update({'crop_box': None, 'center_crop_assumed': False})
    if target != (src_w, src_h) or platform == 'instagram':
        filters.append(f'scale={target[0]}:{target[1]}:flags=lanczos')
    geometry['output_size'] = list(target)

    photo = photo_targets(row, 'video', rng, args)
    if photo['applied'] and (abs(photo['a']-1)>1e-7 or abs(photo['b'])>1e-7):
        # FFmpeg eq brightness/contrast ranges are analogous but not identical
        # to the regression coefficients on decoded/displayed Y. Re-measure.
        filters.append(f"eq=contrast={photo['a']:.7f}:brightness={photo['b']:.7f}")

    in_fps = probe['fps']
    out_fps = fnum(row.get('down_avg_frame_rate'), in_fps) or in_fps
    # Match observed 1-frame leading offset for Kakao 1080p and 2160p.
    drop_frames = max(0, inum(row.get('temporal_offset_frames'), 0))
    if drop_frames and abs(in_fps-out_fps)<0.05:
        filters.append(f'trim=start_frame={drop_frames}')
        filters.append('setpts=PTS-STARTPTS')
    if in_fps > 0 and abs(in_fps-out_fps) >= 0.05:
        filters.append(f'fps=fps={out_fps:.8g}')
    codec = str(row.get('down_codec_name', 'h264')).lower()
    is_hevc = codec in ('hevc', 'h265')
    codec_args = ['-c:v', 'libx265' if is_hevc else 'libx264',
                  '-preset', args.video_preset, '-pix_fmt', 'yuv420p']
    observed_gop = inum(row.get('down_gop_mean'), 150)
    gop = max(1, observed_gop)
    bframes = 2 if flag(row.get('down_has_b_frames')) else 0
    codec_args.extend(['-g', str(gop), '-keyint_min', str(gop),
                       '-sc_threshold', '0', '-bf', str(bframes)])
    profile = str(row.get('down_profile', '')).lower()
    bitrate_target_bps = None
    if is_hevc:
        ratio = fnum(row.get('bitrate_ratio'), 0.3) or 0.3
        ratio *= rng.uniform(1-args.hevc_bitrate_jitter, 1+args.hevc_bitrate_jitter)
        input_bitrate = probe['bitrate_bps'] or None
        if not input_bitrate:
            raise ValueError('HEVC bitrate matching requires input video bitrate from ffprobe')
        bitrate_target_bps = max(100_000, int(input_bitrate * ratio))
        # x264-fitted effective_crf_interp in CSV is NOT a legitimate x265 CRF.
        codec_args.extend(['-b:v', str(bitrate_target_bps),
                           '-tag:v', 'hvc1'])
        if profile in ('main', 'main10'):
            codec_args.extend(['-profile:v', 'main'])
    else:
        ref_crf = fnum(row.get('effective_crf_interp'), 26.0) or 26.0
        crf = clamp(ref_crf + rng.uniform(-args.h264_crf_jitter,
                                           args.h264_crf_jitter), 0, 51)
        codec_args.extend(['-crf', f'{crf:.2f}'])
        if profile in ('high','main','baseline'):
            codec_args.extend(['-profile:v', profile])

    cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
           '-y', '-i', str(src), '-map', '0:v:0', '-map', '0:a?']
    if filters:
        cmd.extend(['-vf', ','.join(filters)])
    cmd += codec_args
    # Explicit output rate stabilizes timing after a one-frame trim.
    cmd += ['-r', f'{out_fps:.8g}', '-fps_mode', 'cfr', '-color_range', 'tv', '-c:a', 'aac',
            '-b:a', '128k', '-movflags', '+faststart', str(dst)]
    try:
        run(cmd)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f'ffmpeg failed: {exc.stderr[-1600:]}') from exc
    out_probe = ffprobe_json(dst) if not args.skip_video_output_probe else None
    return {'verdict': row['reencode_verdict'], 'geometry': geometry,
            'photo': photo, 'video': {
                'codec': codec, 'profile': profile, 'fps': out_fps,
                'input_fps': in_fps, 'reference_gop': gop,
                'bframes': bframes,
                'h264_crf': (None if is_hevc else round(crf, 3)),
                'hevc_target_bitrate_bps': bitrate_target_bps,
                'reference_bitrate_ratio': fnum(row.get('bitrate_ratio')),
                'input_bitrate_bps': probe['bitrate_bps'],
                'output_probe': out_probe,
                'bitrate_match_warning': (
                    'Target bitrate ratio not reached; recalibrate on full clips.'
                    if is_hevc and out_probe and probe['bitrate_bps'] and
                    out_probe['bitrate_bps'] and bitrate_target_bps and
                    abs(out_probe['bitrate_bps'] - bitrate_target_bps)/bitrate_target_bps > 0.30
                    else None),
                'actual_bitrate_ratio': (
                    round(out_probe['bitrate_bps']/probe['bitrate_bps'], 5)
                    if out_probe and probe['bitrate_bps'] and out_probe['bitrate_bps']
                    else None),
                'temporal_offset_frames_used': drop_frames,
                'ffmpeg_filter': ','.join(filters),
            }}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--measurements-dir', type=Path, default=None,
                    help='Deprecated, ignored: policy is embedded (no CSV IO)')
    ap.add_argument('--input', type=Path, help='Train media root (recursive scan)')
    ap.add_argument('--output', type=Path, help='Augmented media root')
    ap.add_argument('--mask-root', type=Path, help='Optional image mask root, same relative paths and .png stem')
    ap.add_argument('--mask-output', type=Path, help='Required when --mask-root is supplied')
    ap.add_argument('--copies', type=int, default=1, help='Independent sampled augmentations per input (default 1)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epoch', type=int, default=0)
    ap.add_argument('--p-clean', type=float, default=0.25)
    ap.add_argument('--p-facebook', type=float, default=0.25)
    ap.add_argument('--p-instagram', type=float, default=0.25)
    ap.add_argument('--p-kakao', type=float, default=0.25)
    ap.add_argument('--p-photo', type=float, default=0.0,
                    help='Apply measured small brightness/contrast target on reencoded items')
    ap.add_argument('--photo-jitter', type=float, default=0.0,
                    help='Relative variation around measured b and (a-1). Proposed, not measured.')
    ap.add_argument('--jpeg-qf-jitter', type=float, default=0.0)
    ap.add_argument('--h264-crf-jitter', type=float, default=0.0)
    ap.add_argument('--hevc-bitrate-jitter', type=float, default=0.0)
    ap.add_argument('--ig-crop-jitter-prob', type=float, default=0.0)
    ap.add_argument('--ig-crop-jitter-frac', type=float, default=0.0,
                    help='Fraction of EXTRA crop margin for displacement')
    ap.add_argument('--video-preset', default='medium', choices=['ultrafast','superfast','veryfast',
                    'faster','fast','medium','slow','slower','veryslow'])
    ap.add_argument('--skip-video-output-probe', action='store_true')
    ap.add_argument('--force-platform', choices=PLATFORMS,
                    help='Debug / ablation: override all platform probabilities')
    ap.add_argument('--force-quality', help='Debug / ablation: force platform quality if valid')
    ap.add_argument('--show-profiles', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='Plan and write manifest but do not transcode or copy')
    ap.add_argument('--continue-on-error', action='store_true')
    return ap.parse_args()


def validate(a: argparse.Namespace):
    if a.show_profiles:
        return
    if not a.input or not a.output:
        raise ValueError('--input and --output are required unless --show-profiles')
    if a.copies <= 0:
        raise ValueError('--copies must be >=1')
    if a.mask_root and not a.mask_output:
        raise ValueError('--mask-root requires --mask-output')
    if (a.mask_root and a.mask_output) and a.mask_root.resolve() == a.mask_output.resolve():
        raise ValueError('mask input and output roots must differ')
    probs = [a.p_clean, a.p_facebook, a.p_instagram, a.p_kakao]
    if min(probs) < 0 or abs(sum(probs) - 1) > 1e-8:
        raise ValueError('Platform probabilities must be >=0 and sum to 1.0')
    if not (0 <= a.p_photo <= 1 and 0 <= a.ig_crop_jitter_prob <= 1):
        raise ValueError('Probabilities must be in [0,1]')
    if any(x < 0 or x > 1 for x in [a.photo_jitter, a.jpeg_qf_jitter/100,
                                   a.hevc_bitrate_jitter, a.ig_crop_jitter_frac]):
        raise ValueError('Invalid jitter; check parameter ranges')
    if a.h264_crf_jitter < 0 or a.h264_crf_jitter > 10:
        raise ValueError('h264_crf_jitter must be between 0 and 10')
    if not a.input.is_dir():
        raise ValueError(f'Input is not a directory: {a.input}')
    if a.output.resolve() == a.input.resolve():
        raise ValueError('Output must be separate from input')
    if a.input.resolve() in a.output.resolve().parents:
        raise ValueError('Do not place output inside input (recursive scan contamination)')


def main():
    args = parse_args()
    policy = MeasurementProfiles()
    if args.show_profiles:
        policy.show()
        return
    validate(args)
    args.output.mkdir(parents=True, exist_ok=True)
    weights = [args.p_clean, args.p_facebook, args.p_instagram, args.p_kakao]
    print('Platform probability:', dict(zip(PLATFORMS, weights)))
    inputs = [p for p in sorted(args.input.rglob('*')) if p.is_file()
              and p.suffix.lower() in (IMAGE_SUFFIXES|VIDEO_SUFFIXES)]
    if not inputs:
        raise RuntimeError(f'No image/video files found in {args.input}')
    manifest = args.output / 'augmentation_manifest.jsonl'
    counts: Counter[str] = Counter()
    with manifest.open('w', encoding='utf8') as writer:
        for i, src in enumerate(inputs, 1):
            rel = src.relative_to(args.input)
            modality = 'image' if src.suffix.lower() in IMAGE_SUFFIXES else 'video'
            try:
                if modality == 'image':
                    from PIL import Image, ImageOps
                    with Image.open(src) as im:
                        # PIL dimensions without orientation changes. Upload
                        # reference sizes are measured after decode; most sources
                        # already have exif orientation=1.
                        trans = ImageOps.exif_transpose(im)
                        w, h = trans.size
                    probe = None
                else:
                    probe = ffprobe_json(src)
                    w, h = probe['width'], probe['height']
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                print('WARN read error', src, exc, file=sys.stderr)
                continue
            band = band_from_dimensions(modality, w, h)
            for copy_index in range(args.copies):
                rng = random.Random(stable_seed(args.seed, rel, copy_index, args.epoch))
                platform = args.force_platform or rng.choices(PLATFORMS, weights, k=1)[0]
                record: dict[str, Any] = {
                    'input': str(src), 'source_relative_path': rel.as_posix(),
                    'copy_index': copy_index, 'seed': args.seed, 'epoch': args.epoch,
                    'modality': modality, 'input_size': [w, h],
                    'input_band': band, 'platform': platform,
                    'conditional_quality_probabilities': (
                        {'high':0.5,'mid':0.25,'low':0.25}
                        if platform=='facebook' and modality=='video' else
                        {'high':1/3,'mid':1/3,'low':1/3}
                        if platform=='facebook' else
                        {'original':1/3,'normal':1/3,'low':1/3}
                        if platform=='kakao' and modality=='image' else {'shared':1.0}
                        if platform=='instagram' else {'single':1.0} if platform=='kakao' else {}),
                }
                try:
                    if platform == 'clean':
                        if args.force_quality:
                            raise ValueError('Cannot force quality on clean route')
                        quality, row = 'none', None
                        out_suffix = src.suffix
                    else:
                        quality = pick_quality(platform, modality, rng, args.force_quality)
                        row, _ = policy.choose(platform, modality, quality, band, w, h, rng)
                        record.update({'quality': quality,
                                       'measured_reference_stimulus_id': row['stimulus_id'],
                                       'measured_reference_target_id': row.get('target_id'),
                                       'measured_reference_verdict': row['reencode_verdict'],
                                       'reference_measurements_csv': 'embedded_aggregate_no_runtime_csv'})
                        verdict = row['reencode_verdict']
                        out_suffix = (src.suffix if modality == 'image' and
                                      verdict == 'container_or_metadata_only' else
                                      '.jpg' if modality == 'image' else '.mp4')
                    dst = args.output / rel.parent / (f'{rel.stem}__aug{copy_index:03d}_'
                                         f'{platform}_{quality}{out_suffix}')
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    record['output'] = str(dst)
                    mask_src = (image_mask_for(src, args.input, args.mask_root)
                                if modality == 'image' and args.mask_root else None)
                    mask_dst = (args.mask_output / rel.parent /
                                f'{rel.stem}__aug{copy_index:03d}_{platform}_{quality}.png'
                                if mask_src and args.mask_output else None)
                    if mask_src:
                        record['input_mask'] = str(mask_src)
                        record['output_mask'] = str(mask_dst)
                    if args.dry_run:
                        record['status'] = 'dry_run'
                    elif platform == 'clean':
                        shutil.copy2(src, dst)
                        if mask_src and mask_dst:
                            mask_dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(mask_src, mask_dst)
                        record['transformation'] = 'bytewise_copy'
                        record['status'] = 'ok'
                    elif modality == 'image':
                        record['transformation'] = transform_image(
                            src, dst, row, platform, rng, args, mask_src, mask_dst)
                        record['status'] = 'ok'
                    else:
                        record['transformation'] = transform_video(
                            src, dst, row, platform, rng, args, probe)
                        record['status'] = 'ok'
                    counts[f'{platform}/{quality}/{band}'] += 1
                except Exception as exc:
                    record['status'] = 'error'
                    record['error'] = str(exc)
                    if not args.continue_on_error:
                        writer.write(json.dumps(record, ensure_ascii=False)+'\n')
                        writer.flush()
                        raise
                    print('WARN transform error', src, str(exc), file=sys.stderr)
                writer.write(json.dumps(record, ensure_ascii=False)+'\n')
        print(f'Processed {len(inputs)} source files, {args.copies} copy/copies each')
    print('Output manifest:', manifest)
    print('Route counts:')
    for key, n in sorted(counts.items()):
        print(f'  {key:34} {n}')
    print('IMPORTANT: Measured reference values do not guarantee identical'
          ' downstream JPEG/codec artifacts. Re-measure sample outputs.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print('ERROR:', e, file=sys.stderr)
        sys.exit(2)