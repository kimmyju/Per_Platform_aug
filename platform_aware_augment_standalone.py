#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CSV-free measurement-derived per-observation platform augmentation.

Version 2: empirically samples observed joint tuples for
(platform, modality, quality, input band, orientation, aspect ratio).
Original experiment rows are embedded without file names, user IDs, or source paths.
No measurements.csv needed at training time. Original row ordinals are within the
source CSV and do not identify training/test media.

Be aware: an observed x264-equivalent CRF, JPEG QF estimate, and codec setting
are *not* exact SNS encoder settings. Outputs must be validated separately.

Photometry is enabled by default on transformed rows: measure codec-only result
then compensate toward the selected observation's measured (a,b), avoiding
blindly stacking the same brightness/contrast changes. Corrections are approximate.
"""

from __future__ import annotations

import argparse
import io
import tempfile
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



# Training mixture 25% each is a proposed sampling distribution, not observed SNS market share.
# Source CSVs retained ONLY as SHA-256 provenance below, not read at runtime.
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

EMPIRICAL_POLICY = {'facebook|image|high': {'le1080/land': [{'a': 1.0, 'b': 0.0, 'bpp': 1.01647, 'crop': False, 'dh': 608, 'dw': 1080, 'icc': True, 'metadata': 'True', 'n': 4, 'oh': 608, 'ow': 1080, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02722, 'crop': False, 'dh': 608, 'dw': 1080, 'icc': True, 'metadata': 'True', 'n': 5, 'oh': 608, 'ow': 1080, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.04087, 'crop': False, 'dh': 608, 'dw': 1080, 'icc': True, 'metadata': 'True', 'n': 6, 'oh': 608, 'ow': 1080, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.03611, 'crop': False, 'dh': 608, 'dw': 1080, 'icc': True, 'metadata': 'True', 'n': 7, 'oh': 608, 'ow': 1080, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'},
                                         {'a': 1.0, 'b': 0.0, 'bpp': 1.036, 'crop': False, 'dh': 608, 'dw': 1080, 'icc': True, 'metadata': 'True', 'n': 8, 'oh': 608, 'ow': 1080, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.03966, 'crop': False, 'dh': 594, 'dw': 1080, 'icc': True, 'metadata': 'True', 'n': 9, 'oh': 594, 'ow': 1080, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.08753, 'crop': False, 'dh': 719, 'dw': 1080, 'icc': True, 'metadata': 'True', 'n': 11, 'oh': 719, 'ow': 1080, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}],
                         'le1080/port': [{'a': 1.0, 'b': 0.0, 'bpp': 1.01986, 'crop': False, 'dh': 1080, 'dw': 844, 'icc': True, 'metadata': 'True', 'n': 1, 'oh': 1080, 'ow': 844, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02661, 'crop': False, 'dh': 1080, 'dw': 720, 'icc': True, 'metadata': 'True', 'n': 2, 'oh': 1080, 'ow': 720, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.0259, 'crop': False, 'dh': 1080, 'dw': 767, 'icc': True, 'metadata': 'True', 'n': 3, 'oh': 1080, 'ow': 767, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}],
                         'le2048/land': [{'a': 0.999829, 'b': -0.000744, 'bpp': 0.2309, 'crop': False, 'dh': 960, 'dw': 1440, 'icc': False, 'metadata': 'True', 'n': 10, 'oh': 1365, 'ow': 2048, 'progressive': True, 'qf': 93.0, 'ss': 0.93688, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000404, 'b': -0.000934, 'bpp': 0.25133, 'crop': False, 'dh': 960, 'dw': 1440, 'icc': False, 'metadata': 'True', 'n': 14, 'oh': 1365, 'ow': 2048, 'progressive': True, 'qf': 93.0, 'ss': 0.95942, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.00058, 'b': -0.001073, 'bpp': 0.22576, 'crop': False, 'dh': 960, 'dw': 1439, 'icc': False, 'metadata': 'True', 'n': 15, 'oh': 1366, 'ow': 2048, 'progressive': True, 'qf': 93.0, 'ss': 0.95432, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.00009, 'b': -0.000855, 'bpp': 0.25523, 'crop': False, 'dh': 960, 'dw': 1440, 'icc': False, 'metadata': 'True', 'n': 18, 'oh': 1365, 'ow': 2048, 'progressive': True, 'qf': 93.0, 'ss': 0.93652, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}],
                         'le2048/port': [{'a': 1.0, 'b': 0.0, 'bpp': 1.01753, 'crop': False, 'dh': 2048, 'dw': 1365, 'icc': True, 'metadata': 'True', 'n': 12, 'oh': 2048, 'ow': 1365, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.06412, 'crop': False, 'dh': 2048, 'dw': 1536, 'icc': True, 'metadata': 'True', 'n': 13, 'oh': 2048, 'ow': 1536, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.23478, 'crop': False, 'dh': 2048, 'dw': 1152, 'icc': True, 'metadata': 'True', 'n': 16, 'oh': 2048, 'ow': 1152, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.04838, 'crop': False, 'dh': 2048, 'dw': 1357, 'icc': True, 'metadata': 'True', 'n': 17, 'oh': 2048, 'ow': 1357, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'},
                                         {'a': 1.0, 'b': 0.0, 'bpp': 1.03826, 'crop': False, 'dh': 2048, 'dw': 1365, 'icc': True, 'metadata': 'True', 'n': 19, 'oh': 2048, 'ow': 1365, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.03747, 'crop': False, 'dh': 2048, 'dw': 1633, 'icc': True, 'metadata': 'True', 'n': 20, 'oh': 2048, 'ow': 1633, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}],
                         'le512/port': [{'a': 1.0, 'b': 0.0, 'bpp': 1.02292, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 21, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02331, 'crop': False, 'dh': 512, 'dw': 444, 'icc': True, 'metadata': 'True', 'n': 22, 'oh': 512, 'ow': 444, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02547, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 23, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02275, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 24, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'},
                                        {'a': 1.0, 'b': 0.0, 'bpp': 1.02307, 'crop': False, 'dh': 512, 'dw': 444, 'icc': True, 'metadata': 'True', 'n': 25, 'oh': 512, 'ow': 444, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.0259, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 26, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02274, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 27, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02283, 'crop': False, 'dh': 512, 'dw': 444, 'icc': True, 'metadata': 'True', 'n': 28, 'oh': 512, 'ow': 444, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'},
                                        {'a': 1.0, 'b': 0.0, 'bpp': 1.02642, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 29, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02293, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 30, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}]},
 'facebook|image|low': {'le1080/land': [{'a': 0.999668, 'b': -0.000599, 'bpp': 0.2951, 'crop': False, 'dh': 360, 'dw': 640, 'icc': False, 'metadata': 'True', 'n': 4, 'oh': 608, 'ow': 1080, 'progressive': True, 'qf': 93.0, 'ss': 0.96641, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 0.999886, 'b': -0.00132, 'bpp': 0.21763, 'crop': False, 'dh': 360, 'dw': 640, 'icc': False, 'metadata': 'True', 'n': 5, 'oh': 608, 'ow': 1080, 'progressive': True, 'qf': 93.0, 'ss': 0.95715, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.00146, 'b': -0.001843, 'bpp': 0.22628, 'crop': False, 'dh': 360, 'dw': 640, 'icc': False, 'metadata': 'True', 'n': 6, 'oh': 608, 'ow': 1080, 'progressive': True, 'qf': 93.0, 'ss': 0.98117, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000328, 'b': -0.001609, 'bpp': 0.263, 'crop': False, 'dh': 360, 'dw': 640, 'icc': False, 'metadata': 'True', 'n': 7, 'oh': 608, 'ow': 1080, 'progressive': True, 'qf': 93.0, 'ss': 0.9853, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'},
                                        {'a': 1.000362, 'b': -0.00164, 'bpp': 0.26287, 'crop': False, 'dh': 360, 'dw': 640, 'icc': False, 'metadata': 'True', 'n': 8, 'oh': 608, 'ow': 1080, 'progressive': True, 'qf': 93.0, 'ss': 0.98533, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000401, 'b': -0.001254, 'bpp': 0.27131, 'crop': False, 'dh': 394, 'dw': 716, 'icc': False, 'metadata': 'True', 'n': 9, 'oh': 594, 'ow': 1080, 'progressive': True, 'qf': 93.0, 'ss': 0.93537, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 0.999708, 'b': -0.001578, 'bpp': 0.29792, 'crop': False, 'dh': 367, 'dw': 552, 'icc': False, 'metadata': 'True', 'n': 11, 'oh': 719, 'ow': 1080, 'progressive': True, 'qf': 93.0, 'ss': 0.92935, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}],
                        'le1080/port': [{'a': 1.001238, 'b': -0.001523, 'bpp': 0.26473, 'crop': False, 'dh': 614, 'dw': 480, 'icc': False, 'metadata': 'True', 'n': 1, 'oh': 1080, 'ow': 844, 'progressive': True, 'qf': 93.0, 'ss': 0.96592, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000126, 'b': -0.002546, 'bpp': 0.22896, 'crop': False, 'dh': 720, 'dw': 480, 'icc': False, 'metadata': 'True', 'n': 2, 'oh': 1080, 'ow': 720, 'progressive': True, 'qf': 93.0, 'ss': 0.95924, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 0.999885, 'b': -0.001124, 'bpp': 0.26676, 'crop': False, 'dh': 676, 'dw': 480, 'icc': False, 'metadata': 'True', 'n': 3, 'oh': 1080, 'ow': 767, 'progressive': True, 'qf': 93.0, 'ss': 0.95571, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}],
                        'le2048/land': [{'a': 1.000142, 'b': -0.000897, 'bpp': 0.33815, 'crop': False, 'dh': 368, 'dw': 552, 'icc': False, 'metadata': 'True', 'n': 10, 'oh': 1365, 'ow': 2048, 'progressive': True, 'qf': 93.0, 'ss': 0.93511, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000784, 'b': -0.001115, 'bpp': 0.42, 'crop': False, 'dh': 368, 'dw': 552, 'icc': False, 'metadata': 'True', 'n': 14, 'oh': 1365, 'ow': 2048, 'progressive': True, 'qf': 93.0, 'ss': 0.94773, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000771, 'b': -0.001102, 'bpp': 0.36246, 'crop': False, 'dh': 368, 'dw': 552, 'icc': False, 'metadata': 'True', 'n': 15, 'oh': 1366, 'ow': 2048, 'progressive': True, 'qf': 93.0, 'ss': 0.95404, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000521, 'b': -0.001012, 'bpp': 0.33532, 'crop': False, 'dh': 368, 'dw': 552, 'icc': False, 'metadata': 'True', 'n': 18, 'oh': 1365, 'ow': 2048, 'progressive': True, 'qf': 93.0, 'ss': 0.9206, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}],
                        'le2048/port': [{'a': 1.000053, 'b': -0.000226, 'bpp': 0.19216, 'crop': False, 'dh': 720, 'dw': 480, 'icc': False, 'metadata': 'True', 'n': 12, 'oh': 2048, 'ow': 1365, 'progressive': True, 'qf': 93.0, 'ss': 0.9608, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000332, 'b': -0.000577, 'bpp': 0.3227, 'crop': False, 'dh': 640, 'dw': 480, 'icc': False, 'metadata': 'True', 'n': 13, 'oh': 2048, 'ow': 1536, 'progressive': True, 'qf': 93.0, 'ss': 0.93428, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000203, 'b': -0.001285, 'bpp': 0.55066, 'crop': False, 'dh': 853, 'dw': 480, 'icc': False, 'metadata': 'True', 'n': 16, 'oh': 2048, 'ow': 1152, 'progressive': True, 'qf': 93.0, 'ss': 0.93841, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 0.999678, 'b': -0.000175, 'bpp': 0.22609, 'crop': False, 'dh': 724, 'dw': 480, 'icc': False, 'metadata': 'True', 'n': 17, 'oh': 2048, 'ow': 1357, 'progressive': True, 'qf': 93.0, 'ss': 0.91125, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'},
                                        {'a': 1.00008, 'b': -0.000264, 'bpp': 0.27113, 'crop': False, 'dh': 720, 'dw': 480, 'icc': False, 'metadata': 'True', 'n': 19, 'oh': 2048, 'ow': 1365, 'progressive': True, 'qf': 93.0, 'ss': 0.93272, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.00037, 'b': -0.000578, 'bpp': 0.36576, 'crop': False, 'dh': 602, 'dw': 480, 'icc': False, 'metadata': 'True', 'n': 20, 'oh': 2048, 'ow': 1633, 'progressive': True, 'qf': 93.0, 'ss': 0.9167, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}],
                        'le512/port': [{'a': 1.0, 'b': 0.0, 'bpp': 1.02292, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 21, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02331, 'crop': False, 'dh': 512, 'dw': 444, 'icc': True, 'metadata': 'True', 'n': 22, 'oh': 512, 'ow': 444, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02547, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 23, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02275, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 24, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'},
                                       {'a': 1.0, 'b': 0.0, 'bpp': 1.02307, 'crop': False, 'dh': 512, 'dw': 444, 'icc': True, 'metadata': 'True', 'n': 25, 'oh': 512, 'ow': 444, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.0259, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 26, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02274, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 27, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02283, 'crop': False, 'dh': 512, 'dw': 444, 'icc': True, 'metadata': 'True', 'n': 28, 'oh': 512, 'ow': 444, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'},
                                       {'a': 1.0, 'b': 0.0, 'bpp': 1.02642, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 29, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02293, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 30, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}]},
 'facebook|image|mid': {'le1080/land': [{'a': 0.999576, 'b': 0.000278, 'bpp': 0.22567, 'crop': False, 'dh': 600, 'dw': 1066, 'icc': False, 'metadata': 'True', 'n': 4, 'oh': 608, 'ow': 1080, 'progressive': True, 'qf': 93.0, 'ss': 0.9739, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 0.999163, 'b': 0.000331, 'bpp': 0.16336, 'crop': False, 'dh': 600, 'dw': 1066, 'icc': False, 'metadata': 'True', 'n': 5, 'oh': 608, 'ow': 1080, 'progressive': True, 'qf': 93.0, 'ss': 0.9549, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 0.999868, 'b': 0.00044, 'bpp': 0.18449, 'crop': False, 'dh': 600, 'dw': 1066, 'icc': False, 'metadata': 'True', 'n': 6, 'oh': 608, 'ow': 1080, 'progressive': True, 'qf': 93.0, 'ss': 0.9755, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 0.998748, 'b': -0.001398, 'bpp': 0.21982, 'crop': False, 'dh': 600, 'dw': 1066, 'icc': False, 'metadata': 'True', 'n': 7, 'oh': 608, 'ow': 1080, 'progressive': True, 'qf': 93.0, 'ss': 0.98089, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'},
                                        {'a': 0.998676, 'b': -0.001345, 'bpp': 0.21989, 'crop': False, 'dh': 600, 'dw': 1066, 'icc': False, 'metadata': 'True', 'n': 8, 'oh': 608, 'ow': 1080, 'progressive': True, 'qf': 93.0, 'ss': 0.98091, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.03966, 'crop': False, 'dh': 594, 'dw': 1080, 'icc': True, 'metadata': 'True', 'n': 9, 'oh': 594, 'ow': 1080, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 0.999392, 'b': -0.000121, 'bpp': 0.28936, 'crop': False, 'dh': 600, 'dw': 901, 'icc': False, 'metadata': 'True', 'n': 11, 'oh': 719, 'ow': 1080, 'progressive': True, 'qf': 93.0, 'ss': 0.94221, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}],
                        'le1080/port': [{'a': 1.00043, 'b': -7e-05, 'bpp': 0.22231, 'crop': False, 'dh': 1079, 'dw': 843, 'icc': False, 'metadata': 'True', 'n': 1, 'oh': 1080, 'ow': 844, 'progressive': True, 'qf': 93.0, 'ss': 0.96777, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02661, 'crop': False, 'dh': 1080, 'dw': 720, 'icc': True, 'metadata': 'True', 'n': 2, 'oh': 1080, 'ow': 720, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.0259, 'crop': False, 'dh': 1080, 'dw': 767, 'icc': True, 'metadata': 'True', 'n': 3, 'oh': 1080, 'ow': 767, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}],
                        'le2048/land': [{'a': 1.000004, 'b': -0.001479, 'bpp': 0.28339, 'crop': False, 'dh': 600, 'dw': 900, 'icc': False, 'metadata': 'True', 'n': 10, 'oh': 1365, 'ow': 2048, 'progressive': True, 'qf': 93.0, 'ss': 0.93839, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000597, 'b': -0.001556, 'bpp': 0.33104, 'crop': False, 'dh': 600, 'dw': 900, 'icc': False, 'metadata': 'True', 'n': 14, 'oh': 1365, 'ow': 2048, 'progressive': True, 'qf': 93.0, 'ss': 0.95405, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000844, 'b': -0.001735, 'bpp': 0.28852, 'crop': False, 'dh': 600, 'dw': 900, 'icc': False, 'metadata': 'True', 'n': 15, 'oh': 1366, 'ow': 2048, 'progressive': True, 'qf': 93.0, 'ss': 0.95521, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000553, 'b': -0.001683, 'bpp': 0.29608, 'crop': False, 'dh': 600, 'dw': 900, 'icc': False, 'metadata': 'True', 'n': 18, 'oh': 1365, 'ow': 2048, 'progressive': True, 'qf': 93.0, 'ss': 0.92786, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}],
                        'le2048/port': [{'a': 1.00005, 'b': -0.001208, 'bpp': 0.13608, 'crop': False, 'dh': 1265, 'dw': 843, 'icc': False, 'metadata': 'True', 'n': 12, 'oh': 2048, 'ow': 1365, 'progressive': True, 'qf': 93.0, 'ss': 0.94741, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.00021, 'b': -0.001765, 'bpp': 0.30131, 'crop': False, 'dh': 1124, 'dw': 843, 'icc': False, 'metadata': 'True', 'n': 13, 'oh': 2048, 'ow': 1536, 'progressive': True, 'qf': 93.0, 'ss': 0.938, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000219, 'b': -0.000783, 'bpp': 0.43762, 'crop': False, 'dh': 1499, 'dw': 843, 'icc': False, 'metadata': 'True', 'n': 16, 'oh': 2048, 'ow': 1152, 'progressive': True, 'qf': 93.0, 'ss': 0.95157, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 0.999861, 'b': -0.001231, 'bpp': 0.20343, 'crop': False, 'dh': 1272, 'dw': 843, 'icc': False, 'metadata': 'True', 'n': 17, 'oh': 2048, 'ow': 1357, 'progressive': True, 'qf': 93.0, 'ss': 0.89132, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'},
                                        {'a': 1.00033, 'b': -0.001334, 'bpp': 0.23925, 'crop': False, 'dh': 1265, 'dw': 843, 'icc': False, 'metadata': 'True', 'n': 19, 'oh': 2048, 'ow': 1365, 'progressive': True, 'qf': 93.0, 'ss': 0.9283, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000821, 'b': -0.002298, 'bpp': 0.29804, 'crop': False, 'dh': 1057, 'dw': 843, 'icc': False, 'metadata': 'True', 'n': 20, 'oh': 2048, 'ow': 1633, 'progressive': True, 'qf': 93.0, 'ss': 0.93583, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}],
                        'le512/port': [{'a': 1.0, 'b': 0.0, 'bpp': 1.02292, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 21, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02331, 'crop': False, 'dh': 512, 'dw': 444, 'icc': True, 'metadata': 'True', 'n': 22, 'oh': 512, 'ow': 444, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02547, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 23, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02275, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 24, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'},
                                       {'a': 1.0, 'b': 0.0, 'bpp': 1.02307, 'crop': False, 'dh': 512, 'dw': 444, 'icc': True, 'metadata': 'True', 'n': 25, 'oh': 512, 'ow': 444, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.0259, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 26, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02274, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 27, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02283, 'crop': False, 'dh': 512, 'dw': 444, 'icc': True, 'metadata': 'True', 'n': 28, 'oh': 512, 'ow': 444, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'},
                                       {'a': 1.0, 'b': 0.0, 'bpp': 1.02642, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 29, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02293, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 30, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}]},
 'facebook|video|high': {'1080p/land': [{'a': 0.997855, 'b': 0.004243, 'bframes': True, 'bitrate_ratio': 0.08606, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 25.19, 'fps': 30.0, 'gop': 150.0, 'n': 1, 'oh': 1080, 'ow': 1920, 'ss': 0.982629, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.998476, 'b': 0.004675, 'bframes': True, 'bitrate_ratio': 0.11218, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 25.63, 'fps': 30.0, 'gop': 150.0, 'n': 2, 'oh': 1080, 'ow': 1920, 'ss': 0.974851, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.999494, 'b': 0.004176, 'bframes': True, 'bitrate_ratio': 0.06227, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 26.63, 'fps': 30.0, 'gop': 150.0, 'n': 3, 'oh': 1080, 'ow': 1920, 'ss': 0.983041, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'},
                                        {'a': 0.999201, 'b': 0.00354, 'bframes': True, 'bitrate_ratio': 0.03722, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 25.13, 'fps': 30.0, 'gop': 150.0, 'n': 4, 'oh': 1080, 'ow': 1920, 'ss': 0.981197, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.995415, 'b': 0.005282, 'bframes': True, 'bitrate_ratio': 0.08232, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 24.72, 'fps': 30.0, 'gop': 150.0, 'n': 5, 'oh': 1080, 'ow': 1920, 'ss': 0.983469, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.999824, 'b': 0.003073, 'bframes': True, 'bitrate_ratio': 0.19061, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 26.0, 'fps': 30.0, 'gop': 150.0, 'n': 13, 'oh': 1080, 'ow': 1920, 'ss': 0.981289, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'},
                                        {'a': 0.997937, 'b': 0.003866, 'bframes': True, 'bitrate_ratio': 0.15718, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 26.73, 'fps': 30.0, 'gop': 150.0, 'n': 15, 'oh': 1080, 'ow': 1920, 'ss': 0.989272, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}],
                         '1080p/port': [{'a': 1.000557, 'b': 0.003164, 'bframes': True, 'bitrate_ratio': 0.18336, 'codec': 'h264', 'crop': False, 'dh': 1280, 'dw': 720, 'effective_crf': 26.0, 'fps': 30.0, 'gop': 150.0, 'n': 11, 'oh': 1920, 'ow': 1080, 'ss': 0.984624, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.999283, 'b': 0.00342, 'bframes': True, 'bitrate_ratio': 0.18666, 'codec': 'h264', 'crop': False, 'dh': 1280, 'dw': 720, 'effective_crf': 26.26, 'fps': 30.0, 'gop': 150.0, 'n': 12, 'oh': 1920, 'ow': 1080, 'ss': 0.970275, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.998939, 'b': 0.003702, 'bframes': True, 'bitrate_ratio': 0.18199, 'codec': 'h264', 'crop': False, 'dh': 1280, 'dw': 720, 'effective_crf': 26.0, 'fps': 30.0, 'gop': 150.0, 'n': 14, 'oh': 1920, 'ow': 1080, 'ss': 0.982357, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}],
                         '2160p/land': [{'a': 0.996082, 'b': 0.004367, 'bframes': True, 'bitrate_ratio': 0.08608, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 26.71, 'fps': 30.0, 'gop': 150.0, 'n': 23, 'oh': 2160, 'ow': 3840, 'ss': 0.959204, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 1.000594, 'b': 0.002627, 'bframes': True, 'bitrate_ratio': 0.06753, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 25.46, 'fps': 30.0, 'gop': 150.0, 'n': 24, 'oh': 2160, 'ow': 3840, 'ss': 0.974739, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 1.001281, 'b': 0.001407, 'bframes': True, 'bitrate_ratio': 0.02546, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 26.53, 'fps': 30.0, 'gop': 150.0, 'n': 25, 'oh': 2160, 'ow': 3840, 'ss': 0.985037, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'},
                                        {'a': 0.998953, 'b': 0.003148, 'bframes': True, 'bitrate_ratio': 0.0515, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 23.67, 'fps': 30.0, 'gop': 150.0, 'n': 27, 'oh': 2160, 'ow': 3840, 'ss': 0.984435, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}],
                         '2160p/port': [{'a': 1.000225, 'b': 0.003691, 'bframes': True, 'bitrate_ratio': 0.05372, 'codec': 'h264', 'crop': False, 'dh': 1280, 'dw': 720, 'effective_crf': 24.49, 'fps': 30.0, 'gop': 150.0, 'n': 21, 'oh': 3840, 'ow': 2160, 'ss': 0.976575, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.997904, 'b': 0.004525, 'bframes': True, 'bitrate_ratio': 0.03513, 'codec': 'h264', 'crop': False, 'dh': 1280, 'dw': 720, 'effective_crf': 25.51, 'fps': 30.0, 'gop': 150.0, 'n': 22, 'oh': 3840, 'ow': 2160, 'ss': 0.976463, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 1.000961, 'b': 0.003267, 'bframes': True, 'bitrate_ratio': 0.05746, 'codec': 'h264', 'crop': False, 'dh': 1276, 'dw': 708, 'effective_crf': 25.23, 'fps': 30.0, 'gop': 150.0, 'n': 26, 'oh': 3840, 'ow': 2132, 'ss': 0.975925, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'},
                                        {'a': 0.999, 'b': 0.004534, 'bframes': True, 'bitrate_ratio': 0.06755, 'codec': 'h264', 'crop': False, 'dh': 1280, 'dw': 720, 'effective_crf': 25.57, 'fps': 30.0, 'gop': 150.0, 'n': 28, 'oh': 3840, 'ow': 2160, 'ss': 0.974344, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.995939, 'b': 0.006364, 'bframes': True, 'bitrate_ratio': 0.06217, 'codec': 'h264', 'crop': False, 'dh': 1280, 'dw': 720, 'effective_crf': 25.0, 'fps': 30.0, 'gop': 150.0, 'n': 29, 'oh': 3840, 'ow': 2160, 'ss': 0.975664, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.998817, 'b': 0.005005, 'bframes': True, 'bitrate_ratio': 0.05838, 'codec': 'h264', 'crop': False, 'dh': 1280, 'dw': 720, 'effective_crf': 26.39, 'fps': 30.0, 'gop': 150.0, 'n': 30, 'oh': 3840, 'ow': 2160, 'ss': 0.979121, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}],
                         '720p/land': [{'a': 0.999126, 'b': 0.001092, 'bframes': True, 'bitrate_ratio': 0.32984, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 25.0, 'fps': 30.0, 'gop': 150.0, 'n': 6, 'oh': 720, 'ow': 1280, 'ss': 0.989328, 'temporal_offset': 0.0, 'verdict': 'recompressed_same_geometry', 'vprofile': 'High'}, {'a': 0.999695, 'b': 0.0007, 'bframes': True, 'bitrate_ratio': 0.35769, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 25.0, 'fps': 30.0, 'gop': 150.0, 'n': 7, 'oh': 720, 'ow': 1280, 'ss': 0.989208, 'temporal_offset': 0.0, 'verdict': 'recompressed_same_geometry', 'vprofile': 'High'}, {'a': 0.999306, 'b': 0.000703, 'bframes': True, 'bitrate_ratio': 0.34625, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 24.8, 'fps': 30.0, 'gop': 150.0, 'n': 8, 'oh': 720, 'ow': 1280, 'ss': 0.99036, 'temporal_offset': 0.0, 'verdict': 'recompressed_same_geometry', 'vprofile': 'High'},
                                       {'a': 1.000939, 'b': 0.000494, 'bframes': True, 'bitrate_ratio': 0.33973, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 24.35, 'fps': 30.0, 'gop': 150.0, 'n': 9, 'oh': 720, 'ow': 1280, 'ss': 0.987625, 'temporal_offset': 0.0, 'verdict': 'recompressed_same_geometry', 'vprofile': 'High'}, {'a': 0.999625, 'b': 0.000779, 'bframes': True, 'bitrate_ratio': 0.32849, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 26.46, 'fps': 30.0, 'gop': 150.0, 'n': 10, 'oh': 720, 'ow': 1280, 'ss': 0.993638, 'temporal_offset': 0.0, 'verdict': 'recompressed_same_geometry', 'vprofile': 'High'}, {'a': 0.999628, 'b': 0.000674, 'bframes': True, 'bitrate_ratio': 0.42224, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 25.0, 'fps': 30.0, 'gop': 150.0, 'n': 16, 'oh': 720, 'ow': 1280, 'ss': 0.989897, 'temporal_offset': 0.0, 'verdict': 'recompressed_same_geometry', 'vprofile': 'High'},
                                       {'a': 0.999135, 'b': 0.001129, 'bframes': True, 'bitrate_ratio': 0.48331, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 23.5, 'fps': 30.0, 'gop': 150.0, 'n': 19, 'oh': 720, 'ow': 1280, 'ss': 0.985556, 'temporal_offset': 0.0, 'verdict': 'recompressed_same_geometry', 'vprofile': 'High'}],
                         '720p/port': [{'a': 1.000151, 'b': 4.3e-05, 'bframes': True, 'bitrate_ratio': 0.3091, 'codec': 'h264', 'crop': False, 'dh': 1280, 'dw': 720, 'effective_crf': 25.49, 'fps': 30.0, 'gop': 150.0, 'n': 17, 'oh': 1280, 'ow': 720, 'ss': 0.993988, 'temporal_offset': 0.0, 'verdict': 'recompressed_same_geometry', 'vprofile': 'High'}, {'a': 0.999758, 'b': 0.000329, 'bframes': True, 'bitrate_ratio': 0.41119, 'codec': 'h264', 'crop': False, 'dh': 1280, 'dw': 720, 'effective_crf': 24.47, 'fps': 30.0, 'gop': 150.0, 'n': 18, 'oh': 1280, 'ow': 720, 'ss': 0.990835, 'temporal_offset': 0.0, 'verdict': 'recompressed_same_geometry', 'vprofile': 'High'}, {'a': 0.999863, 'b': 0.000565, 'bframes': True, 'bitrate_ratio': 0.36243, 'codec': 'h264', 'crop': False, 'dh': 1280, 'dw': 720, 'effective_crf': 25.26, 'fps': 30.0, 'gop': 150.0, 'n': 20, 'oh': 1280, 'ow': 720, 'ss': 0.98624, 'temporal_offset': 0.0, 'verdict': 'recompressed_same_geometry', 'vprofile': 'High'}]},
 'facebook|video|low': {'1080p/land': [{'a': 0.998136, 'b': 0.004363, 'bframes': True, 'bitrate_ratio': 0.01831, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 28.0, 'fps': 30.0, 'gop': 150.0, 'n': 1, 'oh': 1080, 'ow': 1920, 'ss': 0.967417, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.996663, 'b': 0.006493, 'bframes': True, 'bitrate_ratio': 0.02776, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 29.0, 'fps': 30.0, 'gop': 150.0, 'n': 2, 'oh': 1080, 'ow': 1920, 'ss': 0.946399, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.997406, 'b': 0.00653, 'bframes': True, 'bitrate_ratio': 0.01866, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 28.4, 'fps': 30.0, 'gop': 150.0, 'n': 3, 'oh': 1080, 'ow': 1920, 'ss': 0.967217, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'},
                                       {'a': 0.996766, 'b': 0.005297, 'bframes': True, 'bitrate_ratio': 0.00753, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 28.36, 'fps': 30.0, 'gop': 150.0, 'n': 4, 'oh': 1080, 'ow': 1920, 'ss': 0.970287, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.993455, 'b': 0.006896, 'bframes': True, 'bitrate_ratio': 0.01572, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 27.5, 'fps': 30.0, 'gop': 150.0, 'n': 5, 'oh': 1080, 'ow': 1920, 'ss': 0.970694, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.998845, 'b': 0.005237, 'bframes': True, 'bitrate_ratio': 0.0393, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 29.67, 'fps': 30.0, 'gop': 150.0, 'n': 13, 'oh': 1080, 'ow': 1920, 'ss': 0.94915, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'},
                                       {'a': 0.996465, 'b': 0.004721, 'bframes': True, 'bitrate_ratio': 0.04244, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 30.2, 'fps': 30.0, 'gop': 150.0, 'n': 15, 'oh': 1080, 'ow': 1920, 'ss': 0.972872, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}],
                        '1080p/port': [{'a': 0.998769, 'b': 0.005943, 'bframes': True, 'bitrate_ratio': 0.04243, 'codec': 'h264', 'crop': False, 'dh': 640, 'dw': 360, 'effective_crf': 30.0, 'fps': 30.0, 'gop': 150.0, 'n': 11, 'oh': 1920, 'ow': 1080, 'ss': 0.954115, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.994734, 'b': 0.006324, 'bframes': True, 'bitrate_ratio': 0.03879, 'codec': 'h264', 'crop': False, 'dh': 640, 'dw': 360, 'effective_crf': 30.0, 'fps': 30.0, 'gop': 150.0, 'n': 12, 'oh': 1920, 'ow': 1080, 'ss': 0.925748, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.99721, 'b': 0.00504, 'bframes': True, 'bitrate_ratio': 0.04204, 'codec': 'h264', 'crop': False, 'dh': 640, 'dw': 360, 'effective_crf': 29.37, 'fps': 30.0, 'gop': 150.0, 'n': 14, 'oh': 1920, 'ow': 1080, 'ss': 0.94747, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}],
                        '2160p/land': [{'a': 0.987779, 'b': 0.007861, 'bframes': True, 'bitrate_ratio': 0.02008, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 30.72, 'fps': 30.0, 'gop': 150.0, 'n': 23, 'oh': 2160, 'ow': 3840, 'ss': 0.887705, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.998602, 'b': 0.003915, 'bframes': True, 'bitrate_ratio': 0.01246, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 29.0, 'fps': 30.0, 'gop': 150.0, 'n': 24, 'oh': 2160, 'ow': 3840, 'ss': 0.935863, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.997976, 'b': 0.003465, 'bframes': True, 'bitrate_ratio': 0.00598, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 29.71, 'fps': 30.0, 'gop': 150.0, 'n': 25, 'oh': 2160, 'ow': 3840, 'ss': 0.964542, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'},
                                       {'a': 0.998716, 'b': 0.003534, 'bframes': True, 'bitrate_ratio': 0.01266, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 27.22, 'fps': 30.0, 'gop': 150.0, 'n': 27, 'oh': 2160, 'ow': 3840, 'ss': 0.958914, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}],
                        '2160p/port': [{'a': 0.99763, 'b': 0.006514, 'bframes': True, 'bitrate_ratio': 0.00987, 'codec': 'h264', 'crop': False, 'dh': 640, 'dw': 360, 'effective_crf': 28.37, 'fps': 30.0, 'gop': 150.0, 'n': 21, 'oh': 3840, 'ow': 2160, 'ss': 0.934611, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.994142, 'b': 0.007757, 'bframes': True, 'bitrate_ratio': 0.00704, 'codec': 'h264', 'crop': False, 'dh': 640, 'dw': 360, 'effective_crf': 28.38, 'fps': 30.0, 'gop': 150.0, 'n': 22, 'oh': 3840, 'ow': 2160, 'ss': 0.957225, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 1.000003, 'b': 0.003749, 'bframes': True, 'bitrate_ratio': 0.00933, 'codec': 'h264', 'crop': False, 'dh': 634, 'dw': 352, 'effective_crf': 29.0, 'fps': 30.0, 'gop': 150.0, 'n': 26, 'oh': 3840, 'ow': 2132, 'ss': 0.951397, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'},
                                       {'a': 0.99727, 'b': 0.005628, 'bframes': True, 'bitrate_ratio': 0.01544, 'codec': 'h264', 'crop': False, 'dh': 640, 'dw': 360, 'effective_crf': 29.44, 'fps': 30.0, 'gop': 150.0, 'n': 28, 'oh': 3840, 'ow': 2160, 'ss': 0.930212, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.995468, 'b': 0.00802, 'bframes': True, 'bitrate_ratio': 0.01223, 'codec': 'h264', 'crop': False, 'dh': 640, 'dw': 360, 'effective_crf': 29.0, 'fps': 30.0, 'gop': 150.0, 'n': 29, 'oh': 3840, 'ow': 2160, 'ss': 0.939563, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.996821, 'b': 0.007052, 'bframes': True, 'bitrate_ratio': 0.01362, 'codec': 'h264', 'crop': False, 'dh': 640, 'dw': 360, 'effective_crf': 29.74, 'fps': 30.0, 'gop': 150.0, 'n': 30, 'oh': 3840, 'ow': 2160, 'ss': 0.943271, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}],
                        '720p/land': [{'a': 0.996837, 'b': 0.006686, 'bframes': True, 'bitrate_ratio': 0.07079, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 30.65, 'fps': 30.0, 'gop': 150.0, 'n': 6, 'oh': 720, 'ow': 1280, 'ss': 0.96863, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.998193, 'b': 0.004277, 'bframes': True, 'bitrate_ratio': 0.07274, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 30.0, 'fps': 30.0, 'gop': 150.0, 'n': 7, 'oh': 720, 'ow': 1280, 'ss': 0.968892, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.997956, 'b': 0.004734, 'bframes': True, 'bitrate_ratio': 0.07151, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 29.0, 'fps': 30.0, 'gop': 150.0, 'n': 8, 'oh': 720, 'ow': 1280, 'ss': 0.972959, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'},
                                      {'a': 1.005307, 'b': 0.002692, 'bframes': True, 'bitrate_ratio': 0.05935, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 29.0, 'fps': 30.0, 'gop': 150.0, 'n': 9, 'oh': 720, 'ow': 1280, 'ss': 0.96417, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 1.001936, 'b': 0.003395, 'bframes': True, 'bitrate_ratio': 0.0868, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 29.72, 'fps': 30.0, 'gop': 150.0, 'n': 10, 'oh': 720, 'ow': 1280, 'ss': 0.978447, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.998483, 'b': 0.004825, 'bframes': True, 'bitrate_ratio': 0.08903, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 29.54, 'fps': 30.0, 'gop': 150.0, 'n': 16, 'oh': 720, 'ow': 1280, 'ss': 0.971697, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'},
                                      {'a': 0.994607, 'b': 0.006492, 'bframes': True, 'bitrate_ratio': 0.08397, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 29.18, 'fps': 30.0, 'gop': 150.0, 'n': 19, 'oh': 720, 'ow': 1280, 'ss': 0.944548, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}],
                        '720p/port': [{'a': 0.999006, 'b': 0.002316, 'bframes': True, 'bitrate_ratio': 0.07855, 'codec': 'h264', 'crop': False, 'dh': 640, 'dw': 360, 'effective_crf': 27.28, 'fps': 30.0, 'gop': 150.0, 'n': 17, 'oh': 1280, 'ow': 720, 'ss': 0.983888, 'temporal_offset': -1.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.997289, 'b': 0.004237, 'bframes': True, 'bitrate_ratio': 0.0882, 'codec': 'h264', 'crop': False, 'dh': 640, 'dw': 360, 'effective_crf': 29.0, 'fps': 30.0, 'gop': 150.0, 'n': 18, 'oh': 1280, 'ow': 720, 'ss': 0.968041, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.997414, 'b': 0.005795, 'bframes': True, 'bitrate_ratio': 0.07996, 'codec': 'h264', 'crop': False, 'dh': 640, 'dw': 360, 'effective_crf': 29.7, 'fps': 30.0, 'gop': 150.0, 'n': 20, 'oh': 1280, 'ow': 720, 'ss': 0.954918, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}]},
 'facebook|video|mid': {'1080p/land': [{'a': 0.998136, 'b': 0.004363, 'bframes': True, 'bitrate_ratio': 0.01831, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 28.0, 'fps': 30.0, 'gop': 150.0, 'n': 1, 'oh': 1080, 'ow': 1920, 'ss': 0.967417, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.996663, 'b': 0.006493, 'bframes': True, 'bitrate_ratio': 0.02776, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 29.0, 'fps': 30.0, 'gop': 150.0, 'n': 2, 'oh': 1080, 'ow': 1920, 'ss': 0.946399, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.997406, 'b': 0.00653, 'bframes': True, 'bitrate_ratio': 0.01866, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 28.4, 'fps': 30.0, 'gop': 150.0, 'n': 3, 'oh': 1080, 'ow': 1920, 'ss': 0.967217, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'},
                                       {'a': 0.996766, 'b': 0.005297, 'bframes': True, 'bitrate_ratio': 0.00753, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 28.36, 'fps': 30.0, 'gop': 150.0, 'n': 4, 'oh': 1080, 'ow': 1920, 'ss': 0.970287, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.993455, 'b': 0.006896, 'bframes': True, 'bitrate_ratio': 0.01572, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 27.5, 'fps': 30.0, 'gop': 150.0, 'n': 5, 'oh': 1080, 'ow': 1920, 'ss': 0.970694, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.998845, 'b': 0.005237, 'bframes': True, 'bitrate_ratio': 0.0393, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 29.67, 'fps': 30.0, 'gop': 150.0, 'n': 13, 'oh': 1080, 'ow': 1920, 'ss': 0.94915, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'},
                                       {'a': 0.996465, 'b': 0.004721, 'bframes': True, 'bitrate_ratio': 0.04244, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 30.2, 'fps': 30.0, 'gop': 150.0, 'n': 15, 'oh': 1080, 'ow': 1920, 'ss': 0.972872, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}],
                        '1080p/port': [{'a': 0.998769, 'b': 0.005943, 'bframes': True, 'bitrate_ratio': 0.04243, 'codec': 'h264', 'crop': False, 'dh': 640, 'dw': 360, 'effective_crf': 30.0, 'fps': 30.0, 'gop': 150.0, 'n': 11, 'oh': 1920, 'ow': 1080, 'ss': 0.954115, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.994734, 'b': 0.006324, 'bframes': True, 'bitrate_ratio': 0.03879, 'codec': 'h264', 'crop': False, 'dh': 640, 'dw': 360, 'effective_crf': 30.0, 'fps': 30.0, 'gop': 150.0, 'n': 12, 'oh': 1920, 'ow': 1080, 'ss': 0.925748, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.99721, 'b': 0.00504, 'bframes': True, 'bitrate_ratio': 0.04204, 'codec': 'h264', 'crop': False, 'dh': 640, 'dw': 360, 'effective_crf': 29.37, 'fps': 30.0, 'gop': 150.0, 'n': 14, 'oh': 1920, 'ow': 1080, 'ss': 0.94747, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}],
                        '2160p/land': [{'a': 0.987779, 'b': 0.007861, 'bframes': True, 'bitrate_ratio': 0.02008, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 30.72, 'fps': 30.0, 'gop': 150.0, 'n': 23, 'oh': 2160, 'ow': 3840, 'ss': 0.887705, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.998602, 'b': 0.003915, 'bframes': True, 'bitrate_ratio': 0.01246, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 29.0, 'fps': 30.0, 'gop': 150.0, 'n': 24, 'oh': 2160, 'ow': 3840, 'ss': 0.935863, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.997976, 'b': 0.003465, 'bframes': True, 'bitrate_ratio': 0.00598, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 29.71, 'fps': 30.0, 'gop': 150.0, 'n': 25, 'oh': 2160, 'ow': 3840, 'ss': 0.964542, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'},
                                       {'a': 0.998716, 'b': 0.003534, 'bframes': True, 'bitrate_ratio': 0.01266, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 27.22, 'fps': 30.0, 'gop': 150.0, 'n': 27, 'oh': 2160, 'ow': 3840, 'ss': 0.958914, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}],
                        '2160p/port': [{'a': 0.99763, 'b': 0.006514, 'bframes': True, 'bitrate_ratio': 0.00987, 'codec': 'h264', 'crop': False, 'dh': 640, 'dw': 360, 'effective_crf': 28.37, 'fps': 30.0, 'gop': 150.0, 'n': 21, 'oh': 3840, 'ow': 2160, 'ss': 0.934611, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.994142, 'b': 0.007757, 'bframes': True, 'bitrate_ratio': 0.00704, 'codec': 'h264', 'crop': False, 'dh': 640, 'dw': 360, 'effective_crf': 28.38, 'fps': 30.0, 'gop': 150.0, 'n': 22, 'oh': 3840, 'ow': 2160, 'ss': 0.957225, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 1.000003, 'b': 0.003749, 'bframes': True, 'bitrate_ratio': 0.00933, 'codec': 'h264', 'crop': False, 'dh': 634, 'dw': 352, 'effective_crf': 29.0, 'fps': 30.0, 'gop': 150.0, 'n': 26, 'oh': 3840, 'ow': 2132, 'ss': 0.951397, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'},
                                       {'a': 0.99727, 'b': 0.005628, 'bframes': True, 'bitrate_ratio': 0.01544, 'codec': 'h264', 'crop': False, 'dh': 640, 'dw': 360, 'effective_crf': 29.44, 'fps': 30.0, 'gop': 150.0, 'n': 28, 'oh': 3840, 'ow': 2160, 'ss': 0.930212, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.995468, 'b': 0.00802, 'bframes': True, 'bitrate_ratio': 0.01223, 'codec': 'h264', 'crop': False, 'dh': 640, 'dw': 360, 'effective_crf': 29.0, 'fps': 30.0, 'gop': 150.0, 'n': 29, 'oh': 3840, 'ow': 2160, 'ss': 0.939563, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.996821, 'b': 0.007052, 'bframes': True, 'bitrate_ratio': 0.01362, 'codec': 'h264', 'crop': False, 'dh': 640, 'dw': 360, 'effective_crf': 29.74, 'fps': 30.0, 'gop': 150.0, 'n': 30, 'oh': 3840, 'ow': 2160, 'ss': 0.943271, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}],
                        '720p/land': [{'a': 0.996837, 'b': 0.006686, 'bframes': True, 'bitrate_ratio': 0.07079, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 30.65, 'fps': 30.0, 'gop': 150.0, 'n': 6, 'oh': 720, 'ow': 1280, 'ss': 0.96863, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.998193, 'b': 0.004277, 'bframes': True, 'bitrate_ratio': 0.07274, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 30.0, 'fps': 30.0, 'gop': 150.0, 'n': 7, 'oh': 720, 'ow': 1280, 'ss': 0.968892, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.997956, 'b': 0.004734, 'bframes': True, 'bitrate_ratio': 0.07151, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 29.0, 'fps': 30.0, 'gop': 150.0, 'n': 8, 'oh': 720, 'ow': 1280, 'ss': 0.972959, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'},
                                      {'a': 1.005307, 'b': 0.002692, 'bframes': True, 'bitrate_ratio': 0.05935, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 29.0, 'fps': 30.0, 'gop': 150.0, 'n': 9, 'oh': 720, 'ow': 1280, 'ss': 0.96417, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 1.001936, 'b': 0.003395, 'bframes': True, 'bitrate_ratio': 0.0868, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 29.72, 'fps': 30.0, 'gop': 150.0, 'n': 10, 'oh': 720, 'ow': 1280, 'ss': 0.978447, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.998483, 'b': 0.004825, 'bframes': True, 'bitrate_ratio': 0.08903, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 29.54, 'fps': 30.0, 'gop': 150.0, 'n': 16, 'oh': 720, 'ow': 1280, 'ss': 0.971697, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'},
                                      {'a': 0.994607, 'b': 0.006492, 'bframes': True, 'bitrate_ratio': 0.08397, 'codec': 'h264', 'crop': False, 'dh': 360, 'dw': 640, 'effective_crf': 29.18, 'fps': 30.0, 'gop': 150.0, 'n': 19, 'oh': 720, 'ow': 1280, 'ss': 0.944548, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}],
                        '720p/port': [{'a': 0.999006, 'b': 0.002316, 'bframes': True, 'bitrate_ratio': 0.07855, 'codec': 'h264', 'crop': False, 'dh': 640, 'dw': 360, 'effective_crf': 27.28, 'fps': 30.0, 'gop': 150.0, 'n': 17, 'oh': 1280, 'ow': 720, 'ss': 0.983888, 'temporal_offset': -1.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.997289, 'b': 0.004237, 'bframes': True, 'bitrate_ratio': 0.0882, 'codec': 'h264', 'crop': False, 'dh': 640, 'dw': 360, 'effective_crf': 29.0, 'fps': 30.0, 'gop': 150.0, 'n': 18, 'oh': 1280, 'ow': 720, 'ss': 0.968041, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}, {'a': 0.997414, 'b': 0.005795, 'bframes': True, 'bitrate_ratio': 0.07996, 'codec': 'h264', 'crop': False, 'dh': 640, 'dw': 360, 'effective_crf': 29.7, 'fps': 30.0, 'gop': 150.0, 'n': 20, 'oh': 1280, 'ow': 720, 'ss': 0.954918, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Main'}]},
 'instagram|image|shared': {'le1080/land': [{'a': 1.000043, 'b': -3.6e-05, 'bpp': 0.49783, 'crop': True, 'dh': 608, 'dw': 608, 'icc': False, 'metadata': 'True', 'n': 4, 'oh': 608, 'ow': 1080, 'progressive': True, 'qf': 92.0, 'ss': 0.98169, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 0.999849, 'b': -0.002513, 'bpp': 0.37231, 'crop': True, 'dh': 608, 'dw': 608, 'icc': False, 'metadata': 'True', 'n': 5, 'oh': 608, 'ow': 1080, 'progressive': True, 'qf': 92.0, 'ss': 0.96413, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000098, 'b': -7.5e-05, 'bpp': 0.40354, 'crop': True, 'dh': 608, 'dw': 608, 'icc': False, 'metadata': 'True', 'n': 6, 'oh': 608, 'ow': 1080, 'progressive': True, 'qf': 92.0, 'ss': 0.97653, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 0.999998, 'b': 1.4e-05, 'bpp': 0.46685, 'crop': True, 'dh': 608, 'dw': 608, 'icc': False, 'metadata': 'True', 'n': 7, 'oh': 608, 'ow': 1080, 'progressive': True, 'qf': 92.0, 'ss': 0.98834, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'},
                                            {'a': 1.00002, 'b': -3e-06, 'bpp': 0.46685, 'crop': True, 'dh': 608, 'dw': 608, 'icc': False, 'metadata': 'True', 'n': 8, 'oh': 608, 'ow': 1080, 'progressive': True, 'qf': 92.0, 'ss': 0.98835, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000294, 'b': -0.000211, 'bpp': 0.45868, 'crop': True, 'dh': 594, 'dw': 594, 'icc': False, 'metadata': 'True', 'n': 9, 'oh': 594, 'ow': 1080, 'progressive': True, 'qf': 92.0, 'ss': 0.97507, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 0.99943, 'b': -0.002753, 'bpp': 0.41377, 'crop': True, 'dh': 719, 'dw': 719, 'icc': False, 'metadata': 'True', 'n': 11, 'oh': 719, 'ow': 1080, 'progressive': True, 'qf': 92.0, 'ss': 0.929, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}],
                            'le1080/port': [{'a': 0.999953, 'b': -3.7e-05, 'bpp': 0.43061, 'crop': True, 'dh': 844, 'dw': 844, 'icc': False, 'metadata': 'True', 'n': 1, 'oh': 1080, 'ow': 844, 'progressive': True, 'qf': 92.0, 'ss': 0.98614, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000109, 'b': -4.3e-05, 'bpp': 0.41995, 'crop': True, 'dh': 720, 'dw': 720, 'icc': False, 'metadata': 'True', 'n': 2, 'oh': 1080, 'ow': 720, 'progressive': True, 'qf': 92.0, 'ss': 0.9794, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000108, 'b': -5.3e-05, 'bpp': 0.42202, 'crop': True, 'dh': 767, 'dw': 767, 'icc': False, 'metadata': 'True', 'n': 3, 'oh': 1080, 'ow': 767, 'progressive': True, 'qf': 92.0, 'ss': 0.98163, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}],
                            'le2048/land': [{'a': 0.999539, 'b': -0.002652, 'bpp': 0.35007, 'crop': True, 'dh': 1365, 'dw': 1365, 'icc': False, 'metadata': 'True', 'n': 10, 'oh': 1365, 'ow': 2048, 'progressive': True, 'qf': 92.0, 'ss': 0.94857, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000092, 'b': -4.9e-05, 'bpp': 0.41368, 'crop': True, 'dh': 1365, 'dw': 1365, 'icc': False, 'metadata': 'True', 'n': 14, 'oh': 1365, 'ow': 2048, 'progressive': True, 'qf': 92.0, 'ss': 0.9786, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000121, 'b': -5.4e-05, 'bpp': 0.42571, 'crop': True, 'dh': 1366, 'dw': 1366, 'icc': False, 'metadata': 'True', 'n': 15, 'oh': 1366, 'ow': 2048, 'progressive': True, 'qf': 92.0, 'ss': 0.97096, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000085, 'b': -0.002962, 'bpp': 0.3794, 'crop': True, 'dh': 1365, 'dw': 1365, 'icc': False, 'metadata': 'True', 'n': 18, 'oh': 1365, 'ow': 2048, 'progressive': True, 'qf': 92.0, 'ss': 0.95218, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}],
                            'le2048/port': [{'a': 0.999759, 'b': -0.002783, 'bpp': 0.31283, 'crop': True, 'dh': 1365, 'dw': 1365, 'icc': False, 'metadata': 'True', 'n': 12, 'oh': 2048, 'ow': 1365, 'progressive': True, 'qf': 92.0, 'ss': 0.93814, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000013, 'b': -1.3e-05, 'bpp': 0.45439, 'crop': True, 'dh': 1536, 'dw': 1536, 'icc': False, 'metadata': 'True', 'n': 13, 'oh': 2048, 'ow': 1536, 'progressive': True, 'qf': 92.0, 'ss': 0.97236, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000068, 'b': -3.3e-05, 'bpp': 0.49126, 'crop': True, 'dh': 1152, 'dw': 1152, 'icc': False, 'metadata': 'True', 'n': 16, 'oh': 2048, 'ow': 1152, 'progressive': True, 'qf': 92.0, 'ss': 0.99589, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 0.999437, 'b': -0.002683, 'bpp': 0.37251, 'crop': True, 'dh': 1357, 'dw': 1357, 'icc': False, 'metadata': 'True', 'n': 17, 'oh': 2048, 'ow': 1357, 'progressive': True, 'qf': 92.0, 'ss': 0.90092, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'},
                                            {'a': 0.999926, 'b': -0.002843, 'bpp': 0.38018, 'crop': True, 'dh': 1365, 'dw': 1365, 'icc': False, 'metadata': 'True', 'n': 19, 'oh': 2048, 'ow': 1365, 'progressive': True, 'qf': 92.0, 'ss': 0.93683, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 0.999884, 'b': -0.002904, 'bpp': 0.3816, 'crop': True, 'dh': 1633, 'dw': 1633, 'icc': False, 'metadata': 'True', 'n': 20, 'oh': 2048, 'ow': 1633, 'progressive': True, 'qf': 92.0, 'ss': 0.96822, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}],
                            'le512/port': [{'a': 0.999995, 'b': -0.002596, 'bpp': 0.44512, 'crop': True, 'dh': 443, 'dw': 443, 'icc': False, 'metadata': 'True', 'n': 21, 'oh': 512, 'ow': 443, 'progressive': True, 'qf': 92.0, 'ss': 0.97157, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 0.999955, 'b': 3.7e-05, 'bpp': 0.47697, 'crop': True, 'dh': 444, 'dw': 444, 'icc': False, 'metadata': 'True', 'n': 22, 'oh': 512, 'ow': 444, 'progressive': True, 'qf': 92.0, 'ss': 0.98793, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 0.999816, 'b': -0.002356, 'bpp': 0.44424, 'crop': True, 'dh': 443, 'dw': 443, 'icc': False, 'metadata': 'True', 'n': 23, 'oh': 512, 'ow': 443, 'progressive': True, 'qf': 92.0, 'ss': 0.9719, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000001, 'b': -0.002602, 'bpp': 0.44436, 'crop': True, 'dh': 443, 'dw': 443, 'icc': False, 'metadata': 'True', 'n': 24, 'oh': 512, 'ow': 443, 'progressive': True, 'qf': 92.0, 'ss': 0.97153, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'},
                                           {'a': 0.999965, 'b': 2.7e-05, 'bpp': 0.4771, 'crop': True, 'dh': 444, 'dw': 444, 'icc': False, 'metadata': 'True', 'n': 25, 'oh': 512, 'ow': 444, 'progressive': True, 'qf': 92.0, 'ss': 0.98793, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 0.999898, 'b': -0.002431, 'bpp': 0.44471, 'crop': True, 'dh': 443, 'dw': 443, 'icc': False, 'metadata': 'True', 'n': 26, 'oh': 512, 'ow': 443, 'progressive': True, 'qf': 92.0, 'ss': 0.97175, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 0.999988, 'b': -0.002588, 'bpp': 0.44473, 'crop': True, 'dh': 443, 'dw': 443, 'icc': False, 'metadata': 'True', 'n': 27, 'oh': 512, 'ow': 443, 'progressive': True, 'qf': 92.0, 'ss': 0.97153, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 0.999948, 'b': 4.2e-05, 'bpp': 0.47673, 'crop': True, 'dh': 444, 'dw': 444, 'icc': False, 'metadata': 'True', 'n': 28, 'oh': 512, 'ow': 444, 'progressive': True, 'qf': 92.0, 'ss': 0.98794, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'},
                                           {'a': 0.999978, 'b': -0.002499, 'bpp': 0.44549, 'crop': True, 'dh': 443, 'dw': 443, 'icc': False, 'metadata': 'True', 'n': 29, 'oh': 512, 'ow': 443, 'progressive': True, 'qf': 92.0, 'ss': 0.97179, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 0.999999, 'b': -0.002598, 'bpp': 0.44465, 'crop': True, 'dh': 443, 'dw': 443, 'icc': False, 'metadata': 'True', 'n': 30, 'oh': 512, 'ow': 443, 'progressive': True, 'qf': 92.0, 'ss': 0.97152, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}]},
 'instagram|video|shared': {'1080p/land': [{'a': 0.995396, 'b': 0.005321, 'bframes': True, 'bitrate_ratio': 0.06523, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 25.24, 'fps': 30.0, 'gop': 150.0, 'n': 1, 'oh': 1080, 'ow': 1920, 'ss': 0.978971, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.998726, 'b': 0.005145, 'bframes': True, 'bitrate_ratio': 0.07671, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 25.73, 'fps': 30.0, 'gop': 150.0, 'n': 2, 'oh': 1080, 'ow': 1920, 'ss': 0.969611, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.998952, 'b': 0.005192, 'bframes': True, 'bitrate_ratio': 0.03998, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 26.37, 'fps': 30.0, 'gop': 150.0, 'n': 3, 'oh': 1080, 'ow': 1920, 'ss': 0.981676, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'},
                                           {'a': 0.998368, 'b': 0.0045, 'bframes': True, 'bitrate_ratio': 0.03612, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 24.77, 'fps': 30.0, 'gop': 150.0, 'n': 4, 'oh': 1080, 'ow': 1920, 'ss': 0.974793, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.992149, 'b': 0.006437, 'bframes': True, 'bitrate_ratio': 0.06102, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 24.73, 'fps': 30.0, 'gop': 150.0, 'n': 5, 'oh': 1080, 'ow': 1920, 'ss': 0.980079, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 1.000168, 'b': 0.00343, 'bframes': True, 'bitrate_ratio': 0.11773, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 26.0, 'fps': 30.0, 'gop': 150.0, 'n': 13, 'oh': 1080, 'ow': 1920, 'ss': 0.977987, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'},
                                           {'a': 0.997748, 'b': 0.004342, 'bframes': True, 'bitrate_ratio': 0.11126, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 27.0, 'fps': 30.0, 'gop': 150.0, 'n': 15, 'oh': 1080, 'ow': 1920, 'ss': 0.986999, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}],
                            '1080p/port': [{'a': 0.999583, 'b': 0.003807, 'bframes': True, 'bitrate_ratio': 0.10651, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 26.0, 'fps': 30.0, 'gop': 150.0, 'n': 11, 'oh': 1920, 'ow': 1080, 'ss': 0.983593, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.999256, 'b': 0.00363, 'bframes': True, 'bitrate_ratio': 0.12955, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 26.25, 'fps': 30.0, 'gop': 150.0, 'n': 12, 'oh': 1920, 'ow': 1080, 'ss': 0.965628, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.99881, 'b': 0.004112, 'bframes': True, 'bitrate_ratio': 0.11374, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 26.35, 'fps': 30.0, 'gop': 150.0, 'n': 14, 'oh': 1920, 'ow': 1080, 'ss': 0.97942, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}],
                            '2160p/land': [{'a': 0.996493, 'b': 0.004516, 'bframes': True, 'bitrate_ratio': 0.04614, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 27.0, 'fps': 30.0, 'gop': 150.0, 'n': 23, 'oh': 2160, 'ow': 3840, 'ss': 0.959453, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 1.001927, 'b': 0.00269, 'bframes': True, 'bitrate_ratio': 0.03603, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 25.37, 'fps': 30.0, 'gop': 150.0, 'n': 24, 'oh': 2160, 'ow': 3840, 'ss': 0.975346, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.999907, 'b': 0.002615, 'bframes': True, 'bitrate_ratio': 0.01904, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 26.29, 'fps': 30.0, 'gop': 150.0, 'n': 25, 'oh': 2160, 'ow': 3840, 'ss': 0.981997, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'},
                                           {'a': 0.999084, 'b': 0.003199, 'bframes': True, 'bitrate_ratio': 0.0307, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 24.0, 'fps': 30.0, 'gop': 150.0, 'n': 27, 'oh': 2160, 'ow': 3840, 'ss': 0.983601, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}],
                            '2160p/port': [{'a': 0.999081, 'b': 0.004667, 'bframes': True, 'bitrate_ratio': 0.02822, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 24.54, 'fps': 30.0, 'gop': 150.0, 'n': 21, 'oh': 3840, 'ow': 2160, 'ss': 0.97516, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.997787, 'b': 0.004827, 'bframes': True, 'bitrate_ratio': 0.02059, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 26.13, 'fps': 30.0, 'gop': 150.0, 'n': 22, 'oh': 3840, 'ow': 2160, 'ss': 0.977239, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 1.001272, 'b': 0.002747, 'bframes': True, 'bitrate_ratio': 0.03622, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 25.42, 'fps': 30.0, 'gop': 150.0, 'n': 26, 'oh': 3840, 'ow': 2132, 'ss': 0.975905, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'},
                                           {'a': 0.998627, 'b': 0.004629, 'bframes': True, 'bitrate_ratio': 0.04267, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 25.74, 'fps': 30.0, 'gop': 150.0, 'n': 28, 'oh': 3840, 'ow': 2160, 'ss': 0.97142, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.994101, 'b': 0.00694, 'bframes': True, 'bitrate_ratio': 0.0332, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 25.0, 'fps': 30.0, 'gop': 150.0, 'n': 29, 'oh': 3840, 'ow': 2160, 'ss': 0.974801, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.999044, 'b': 0.005049, 'bframes': True, 'bitrate_ratio': 0.0353, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 26.63, 'fps': 30.0, 'gop': 150.0, 'n': 30, 'oh': 3840, 'ow': 2160, 'ss': 0.976907, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}],
                            '720p/land': [{'a': 0.999281, 'b': 0.001835, 'bframes': True, 'bitrate_ratio': 0.2441, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 24.5, 'fps': 30.0, 'gop': 150.0, 'n': 6, 'oh': 720, 'ow': 1280, 'ss': 0.986142, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.999626, 'b': 0.000967, 'bframes': True, 'bitrate_ratio': 0.2474, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 25.0, 'fps': 30.0, 'gop': 150.0, 'n': 7, 'oh': 720, 'ow': 1280, 'ss': 0.985944, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.999051, 'b': 0.001093, 'bframes': True, 'bitrate_ratio': 0.26796, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 24.7, 'fps': 30.0, 'gop': 150.0, 'n': 8, 'oh': 720, 'ow': 1280, 'ss': 0.986674, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'},
                                          {'a': 1.001734, 'b': 0.000621, 'bframes': True, 'bitrate_ratio': 0.19952, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 24.49, 'fps': 30.0, 'gop': 150.0, 'n': 9, 'oh': 720, 'ow': 1280, 'ss': 0.985409, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.999769, 'b': 0.001125, 'bframes': True, 'bitrate_ratio': 0.19472, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 26.0, 'fps': 30.0, 'gop': 150.0, 'n': 10, 'oh': 720, 'ow': 1280, 'ss': 0.992458, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.999147, 'b': 0.001138, 'bframes': True, 'bitrate_ratio': 0.28167, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 25.27, 'fps': 30.0, 'gop': 150.0, 'n': 16, 'oh': 720, 'ow': 1280, 'ss': 0.986927, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'},
                                          {'a': 0.998116, 'b': 0.001422, 'bframes': True, 'bitrate_ratio': 0.27667, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 24.0, 'fps': 30.0, 'gop': 150.0, 'n': 19, 'oh': 720, 'ow': 1280, 'ss': 0.982199, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}],
                            '720p/port': [{'a': 1.000173, 'b': 0.000218, 'bframes': True, 'bitrate_ratio': 0.22308, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 26.0, 'fps': 30.0, 'gop': 150.0, 'n': 17, 'oh': 1280, 'ow': 720, 'ss': 0.990133, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.999512, 'b': 0.00083, 'bframes': True, 'bitrate_ratio': 0.24187, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 25.0, 'fps': 30.0, 'gop': 150.0, 'n': 18, 'oh': 1280, 'ow': 720, 'ss': 0.988422, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}, {'a': 0.999798, 'b': 0.000618, 'bframes': True, 'bitrate_ratio': 0.22879, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 720, 'effective_crf': 25.51, 'fps': 30.0, 'gop': 150.0, 'n': 20, 'oh': 1280, 'ow': 720, 'ss': 0.981634, 'temporal_offset': 0.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'High'}]},
 'kakao|image|low': {'le1080/land': [{'a': 1.003752, 'b': -0.00356, 'bpp': 0.3212, 'crop': False, 'dh': 541, 'dw': 960, 'icc': False, 'metadata': 'True', 'n': 4, 'oh': 608, 'ow': 1080, 'progressive': False, 'qf': 86.0, 'ss': 0.97498, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.00112, 'b': -0.004558, 'bpp': 0.23457, 'crop': False, 'dh': 541, 'dw': 960, 'icc': False, 'metadata': 'True', 'n': 5, 'oh': 608, 'ow': 1080, 'progressive': False, 'qf': 86.0, 'ss': 0.95895, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 0.99854, 'b': -0.001915, 'bpp': 0.29722, 'crop': False, 'dh': 541, 'dw': 960, 'icc': False, 'metadata': 'True', 'n': 6, 'oh': 608, 'ow': 1080, 'progressive': False, 'qf': 86.0, 'ss': 0.9788, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 0.99851, 'b': -0.002912, 'bpp': 0.35276, 'crop': False, 'dh': 541, 'dw': 960, 'icc': False, 'metadata': 'True', 'n': 7, 'oh': 608, 'ow': 1080, 'progressive': False, 'qf': 86.0, 'ss': 0.98577, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'},
                                     {'a': 0.998495, 'b': -0.002911, 'bpp': 0.3527, 'crop': False, 'dh': 541, 'dw': 960, 'icc': False, 'metadata': 'True', 'n': 8, 'oh': 608, 'ow': 1080, 'progressive': False, 'qf': 86.0, 'ss': 0.98577, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000677, 'b': -0.002356, 'bpp': 0.32782, 'crop': False, 'dh': 528, 'dw': 960, 'icc': False, 'metadata': 'True', 'n': 9, 'oh': 594, 'ow': 1080, 'progressive': False, 'qf': 86.0, 'ss': 0.95458, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.004861, 'b': -0.006468, 'bpp': 0.36119, 'crop': False, 'dh': 640, 'dw': 960, 'icc': False, 'metadata': 'True', 'n': 11, 'oh': 719, 'ow': 1080, 'progressive': False, 'qf': 86.0, 'ss': 0.94984, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}],
                     'le1080/port': [{'a': 0.999176, 'b': -0.001474, 'bpp': 0.32377, 'crop': False, 'dh': 960, 'dw': 751, 'icc': False, 'metadata': 'True', 'n': 1, 'oh': 1080, 'ow': 844, 'progressive': False, 'qf': 86.0, 'ss': 0.97284, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.002254, 'b': -0.005319, 'bpp': 0.29458, 'crop': False, 'dh': 960, 'dw': 640, 'icc': False, 'metadata': 'True', 'n': 2, 'oh': 1080, 'ow': 720, 'progressive': False, 'qf': 86.0, 'ss': 0.96584, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.001218, 'b': -0.002514, 'bpp': 0.31417, 'crop': False, 'dh': 960, 'dw': 682, 'icc': False, 'metadata': 'True', 'n': 3, 'oh': 1080, 'ow': 767, 'progressive': False, 'qf': 86.0, 'ss': 0.96871, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}],
                     'le2048/land': [{'a': 0.998875, 'b': -0.004826, 'bpp': 0.35155, 'crop': False, 'dh': 640, 'dw': 960, 'icc': False, 'metadata': 'True', 'n': 10, 'oh': 1365, 'ow': 2048, 'progressive': False, 'qf': 86.0, 'ss': 0.93443, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 0.999956, 'b': -0.003494, 'bpp': 0.41435, 'crop': False, 'dh': 640, 'dw': 960, 'icc': False, 'metadata': 'True', 'n': 14, 'oh': 1365, 'ow': 2048, 'progressive': False, 'qf': 86.0, 'ss': 0.95578, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.001358, 'b': -0.00599, 'bpp': 0.3693, 'crop': False, 'dh': 641, 'dw': 960, 'icc': False, 'metadata': 'True', 'n': 15, 'oh': 1366, 'ow': 2048, 'progressive': False, 'qf': 86.0, 'ss': 0.95553, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000843, 'b': -0.005789, 'bpp': 0.36525, 'crop': False, 'dh': 640, 'dw': 960, 'icc': False, 'metadata': 'True', 'n': 18, 'oh': 1365, 'ow': 2048, 'progressive': False, 'qf': 86.0, 'ss': 0.92453, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}],
                     'le2048/port': [{'a': 1.000222, 'b': -0.003693, 'bpp': 0.21099, 'crop': False, 'dh': 960, 'dw': 640, 'icc': False, 'metadata': 'True', 'n': 12, 'oh': 2048, 'ow': 1365, 'progressive': False, 'qf': 86.0, 'ss': 0.95611, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000947, 'b': -0.004107, 'bpp': 0.39871, 'crop': False, 'dh': 960, 'dw': 720, 'icc': False, 'metadata': 'True', 'n': 13, 'oh': 2048, 'ow': 1536, 'progressive': False, 'qf': 86.0, 'ss': 0.93536, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000785, 'b': -0.004431, 'bpp': 0.69222, 'crop': False, 'dh': 960, 'dw': 540, 'icc': False, 'metadata': 'True', 'n': 16, 'oh': 2048, 'ow': 1152, 'progressive': False, 'qf': 86.0, 'ss': 0.94204, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.001259, 'b': -0.004304, 'bpp': 0.28618, 'crop': False, 'dh': 960, 'dw': 637, 'icc': False, 'metadata': 'True', 'n': 17, 'oh': 2048, 'ow': 1357, 'progressive': False, 'qf': 86.0, 'ss': 0.90497, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'},
                                     {'a': 1.001774, 'b': -0.00458, 'bpp': 0.3366, 'crop': False, 'dh': 960, 'dw': 640, 'icc': False, 'metadata': 'True', 'n': 19, 'oh': 2048, 'ow': 1365, 'progressive': False, 'qf': 86.0, 'ss': 0.9332, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.001165, 'b': -0.004119, 'bpp': 0.40281, 'crop': False, 'dh': 960, 'dw': 766, 'icc': False, 'metadata': 'True', 'n': 20, 'oh': 2048, 'ow': 1633, 'progressive': False, 'qf': 86.0, 'ss': 0.93368, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}],
                     'le512/port': [{'a': 0.999534, 'b': -0.00097, 'bpp': 0.36606, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 21, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 86.0, 'ss': 0.9786, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}, {'a': 0.999625, 'b': -0.001047, 'bpp': 0.36287, 'crop': False, 'dh': 512, 'dw': 444, 'icc': True, 'metadata': 'True', 'n': 22, 'oh': 512, 'ow': 444, 'progressive': False, 'qf': 86.0, 'ss': 0.97921, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}, {'a': 0.998902, 'b': -0.001016, 'bpp': 0.36749, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 23, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 86.0, 'ss': 0.97934, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}, {'a': 0.999525, 'b': -0.000969, 'bpp': 0.36556, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 24, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 86.0, 'ss': 0.97859, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'},
                                    {'a': 0.999623, 'b': -0.001048, 'bpp': 0.36276, 'crop': False, 'dh': 512, 'dw': 444, 'icc': True, 'metadata': 'True', 'n': 25, 'oh': 512, 'ow': 444, 'progressive': False, 'qf': 86.0, 'ss': 0.97922, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}, {'a': 0.998826, 'b': -0.000944, 'bpp': 0.36776, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 26, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 86.0, 'ss': 0.97932, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}, {'a': 0.999491, 'b': -0.000943, 'bpp': 0.36563, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 27, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 86.0, 'ss': 0.97861, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}, {'a': 0.999638, 'b': -0.001057, 'bpp': 0.36268, 'crop': False, 'dh': 512, 'dw': 444, 'icc': True, 'metadata': 'True', 'n': 28, 'oh': 512, 'ow': 444, 'progressive': False, 'qf': 86.0, 'ss': 0.97923, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'},
                                    {'a': 0.998712, 'b': -0.000851, 'bpp': 0.36842, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 29, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 86.0, 'ss': 0.97934, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}, {'a': 0.999536, 'b': -0.000973, 'bpp': 0.36558, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 30, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 86.0, 'ss': 0.9786, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}]},
 'kakao|image|normal': {'le1080/land': [{'a': 1.003085, 'b': -0.003202, 'bpp': 0.61062, 'crop': False, 'dh': 608, 'dw': 1080, 'icc': True, 'metadata': 'True', 'n': 4, 'oh': 608, 'ow': 1080, 'progressive': False, 'qf': 96.0, 'ss': 0.98648, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}, {'a': 1.000349, 'b': -0.001978, 'bpp': 0.49212, 'crop': False, 'dh': 608, 'dw': 1080, 'icc': True, 'metadata': 'True', 'n': 5, 'oh': 608, 'ow': 1080, 'progressive': False, 'qf': 96.0, 'ss': 0.96761, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}, {'a': 0.999924, 'b': -0.001678, 'bpp': 0.57466, 'crop': False, 'dh': 608, 'dw': 1080, 'icc': True, 'metadata': 'True', 'n': 6, 'oh': 608, 'ow': 1080, 'progressive': False, 'qf': 96.0, 'ss': 0.98569, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}, {'a': 0.998993, 'b': -0.00106, 'bpp': 0.66471, 'crop': False, 'dh': 608, 'dw': 1080, 'icc': True, 'metadata': 'True', 'n': 7, 'oh': 608, 'ow': 1080, 'progressive': False, 'qf': 96.0, 'ss': 0.99284, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'},
                                        {'a': 0.99899, 'b': -0.001055, 'bpp': 0.66481, 'crop': False, 'dh': 608, 'dw': 1080, 'icc': True, 'metadata': 'True', 'n': 8, 'oh': 608, 'ow': 1080, 'progressive': False, 'qf': 96.0, 'ss': 0.99285, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}, {'a': 1.000332, 'b': -0.001885, 'bpp': 0.63433, 'crop': False, 'dh': 594, 'dw': 1080, 'icc': True, 'metadata': 'True', 'n': 9, 'oh': 594, 'ow': 1080, 'progressive': False, 'qf': 96.0, 'ss': 0.98304, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}, {'a': 1.000421, 'b': -0.001919, 'bpp': 0.64848, 'crop': False, 'dh': 719, 'dw': 1080, 'icc': True, 'metadata': 'True', 'n': 11, 'oh': 719, 'ow': 1080, 'progressive': False, 'qf': 96.0, 'ss': 0.98526, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}],
                        'le1080/port': [{'a': 0.998892, 'b': -0.001127, 'bpp': 0.63923, 'crop': False, 'dh': 1080, 'dw': 844, 'icc': True, 'metadata': 'True', 'n': 1, 'oh': 1080, 'ow': 844, 'progressive': False, 'qf': 96.0, 'ss': 0.98895, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}, {'a': 1.000278, 'b': -0.001866, 'bpp': 0.59533, 'crop': False, 'dh': 1080, 'dw': 720, 'icc': True, 'metadata': 'True', 'n': 2, 'oh': 1080, 'ow': 720, 'progressive': False, 'qf': 96.0, 'ss': 0.98434, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}, {'a': 1.000719, 'b': -0.002071, 'bpp': 0.61649, 'crop': False, 'dh': 1080, 'dw': 767, 'icc': True, 'metadata': 'True', 'n': 3, 'oh': 1080, 'ow': 767, 'progressive': False, 'qf': 96.0, 'ss': 0.9845, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}],
                        'le2048/land': [{'a': 0.999186, 'b': -0.004199, 'bpp': 0.58603, 'crop': False, 'dh': 960, 'dw': 1440, 'icc': False, 'metadata': 'True', 'n': 10, 'oh': 1365, 'ow': 2048, 'progressive': False, 'qf': 96.0, 'ss': 0.96382, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.0, 'b': -0.00262, 'bpp': 0.63831, 'crop': False, 'dh': 960, 'dw': 1440, 'icc': False, 'metadata': 'True', 'n': 14, 'oh': 1365, 'ow': 2048, 'progressive': False, 'qf': 96.0, 'ss': 0.97953, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000578, 'b': -0.002741, 'bpp': 0.60204, 'crop': False, 'dh': 961, 'dw': 1440, 'icc': False, 'metadata': 'True', 'n': 15, 'oh': 1366, 'ow': 2048, 'progressive': False, 'qf': 96.0, 'ss': 0.97762, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000511, 'b': -0.004756, 'bpp': 0.59126, 'crop': False, 'dh': 960, 'dw': 1440, 'icc': False, 'metadata': 'True', 'n': 18, 'oh': 1365, 'ow': 2048, 'progressive': False, 'qf': 96.0, 'ss': 0.96237, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}],
                        'le2048/port': [{'a': 0.999789, 'b': -0.002599, 'bpp': 0.44772, 'crop': False, 'dh': 1440, 'dw': 960, 'icc': False, 'metadata': 'True', 'n': 12, 'oh': 2048, 'ow': 1365, 'progressive': False, 'qf': 96.0, 'ss': 0.96285, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000487, 'b': -0.002759, 'bpp': 0.64644, 'crop': False, 'dh': 1440, 'dw': 1080, 'icc': False, 'metadata': 'True', 'n': 13, 'oh': 2048, 'ow': 1536, 'progressive': False, 'qf': 96.0, 'ss': 0.97361, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000587, 'b': -0.002861, 'bpp': 1.0617, 'crop': False, 'dh': 1440, 'dw': 810, 'icc': False, 'metadata': 'True', 'n': 16, 'oh': 2048, 'ow': 1152, 'progressive': False, 'qf': 96.0, 'ss': 0.97993, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.000734, 'b': -0.003148, 'bpp': 0.51556, 'crop': False, 'dh': 1440, 'dw': 955, 'icc': False, 'metadata': 'True', 'n': 17, 'oh': 2048, 'ow': 1357, 'progressive': False, 'qf': 96.0, 'ss': 0.9406, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'},
                                        {'a': 1.001309, 'b': -0.003425, 'bpp': 0.58922, 'crop': False, 'dh': 1440, 'dw': 960, 'icc': False, 'metadata': 'True', 'n': 19, 'oh': 2048, 'ow': 1365, 'progressive': False, 'qf': 96.0, 'ss': 0.96688, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}, {'a': 1.001233, 'b': -0.003328, 'bpp': 0.62815, 'crop': False, 'dh': 1440, 'dw': 1148, 'icc': False, 'metadata': 'True', 'n': 20, 'oh': 2048, 'ow': 1633, 'progressive': False, 'qf': 96.0, 'ss': 0.97136, 'subsamp': '4:2:0', 'verdict': 'resized_and_recompressed'}],
                        'le512/port': [{'a': 0.999782, 'b': -0.001098, 'bpp': 0.66983, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 21, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 96.0, 'ss': 0.99071, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}, {'a': 0.999764, 'b': -0.001111, 'bpp': 0.66479, 'crop': False, 'dh': 512, 'dw': 444, 'icc': True, 'metadata': 'True', 'n': 22, 'oh': 512, 'ow': 444, 'progressive': False, 'qf': 96.0, 'ss': 0.99072, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}, {'a': 0.998921, 'b': -0.000929, 'bpp': 0.67292, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 23, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 96.0, 'ss': 0.99075, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}, {'a': 0.999798, 'b': -0.001116, 'bpp': 0.66903, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 24, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 96.0, 'ss': 0.99069, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'},
                                       {'a': 0.999761, 'b': -0.00111, 'bpp': 0.66484, 'crop': False, 'dh': 512, 'dw': 444, 'icc': True, 'metadata': 'True', 'n': 25, 'oh': 512, 'ow': 444, 'progressive': False, 'qf': 96.0, 'ss': 0.99072, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}, {'a': 0.998757, 'b': -0.000845, 'bpp': 0.67276, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 26, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 96.0, 'ss': 0.99071, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}, {'a': 0.999774, 'b': -0.001096, 'bpp': 0.66942, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 27, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 96.0, 'ss': 0.9907, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}, {'a': 0.999773, 'b': -0.001116, 'bpp': 0.66439, 'crop': False, 'dh': 512, 'dw': 444, 'icc': True, 'metadata': 'True', 'n': 28, 'oh': 512, 'ow': 444, 'progressive': False, 'qf': 96.0, 'ss': 0.99073, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'},
                                       {'a': 0.998775, 'b': -0.000833, 'bpp': 0.67373, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 29, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 96.0, 'ss': 0.99072, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}, {'a': 0.999794, 'b': -0.001112, 'bpp': 0.66907, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'True', 'n': 30, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 96.0, 'ss': 0.99069, 'subsamp': '4:2:0', 'verdict': 'recompressed_same_geometry'}]},
 'kakao|image|original': {'le1080/land': [{'a': 1.0, 'b': 0.0, 'bpp': 1.02746, 'crop': False, 'dh': 608, 'dw': 1080, 'icc': True, 'metadata': 'False', 'n': 4, 'oh': 608, 'ow': 1080, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02319, 'crop': False, 'dh': 608, 'dw': 1080, 'icc': True, 'metadata': 'False', 'n': 5, 'oh': 608, 'ow': 1080, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.01764, 'crop': False, 'dh': 608, 'dw': 1080, 'icc': True, 'metadata': 'False', 'n': 6, 'oh': 608, 'ow': 1080, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02079, 'crop': False, 'dh': 608, 'dw': 1080, 'icc': True, 'metadata': 'False', 'n': 7, 'oh': 608, 'ow': 1080, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'},
                                          {'a': 1.0, 'b': 0.0, 'bpp': 1.02079, 'crop': False, 'dh': 608, 'dw': 1080, 'icc': True, 'metadata': 'False', 'n': 8, 'oh': 608, 'ow': 1080, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.01915, 'crop': False, 'dh': 594, 'dw': 1080, 'icc': True, 'metadata': 'False', 'n': 9, 'oh': 594, 'ow': 1080, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.01585, 'crop': False, 'dh': 719, 'dw': 1080, 'icc': True, 'metadata': 'False', 'n': 11, 'oh': 719, 'ow': 1080, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}],
                          'le1080/port': [{'a': 1.0, 'b': 0.0, 'bpp': 1.025, 'crop': False, 'dh': 1080, 'dw': 844, 'icc': True, 'metadata': 'False', 'n': 1, 'oh': 1080, 'ow': 844, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.01979, 'crop': False, 'dh': 1080, 'dw': 720, 'icc': True, 'metadata': 'False', 'n': 2, 'oh': 1080, 'ow': 720, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.02675, 'crop': False, 'dh': 1080, 'dw': 767, 'icc': True, 'metadata': 'False', 'n': 3, 'oh': 1080, 'ow': 767, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}],
                          'le2048/land': [{'a': 1.0, 'b': 0.0, 'bpp': 1.00584, 'crop': False, 'dh': 1365, 'dw': 2048, 'icc': True, 'metadata': 'False', 'n': 10, 'oh': 1365, 'ow': 2048, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.00888, 'crop': False, 'dh': 1365, 'dw': 2048, 'icc': True, 'metadata': 'False', 'n': 14, 'oh': 1365, 'ow': 2048, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.00775, 'crop': False, 'dh': 1366, 'dw': 2048, 'icc': True, 'metadata': 'False', 'n': 15, 'oh': 1366, 'ow': 2048, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.00601, 'crop': False, 'dh': 1365, 'dw': 2048, 'icc': True, 'metadata': 'False', 'n': 18, 'oh': 1365, 'ow': 2048, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}],
                          'le2048/port': [{'a': 1.0, 'b': 0.0, 'bpp': 1.00636, 'crop': False, 'dh': 2048, 'dw': 1365, 'icc': True, 'metadata': 'False', 'n': 12, 'oh': 2048, 'ow': 1365, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.00479, 'crop': False, 'dh': 2048, 'dw': 1536, 'icc': True, 'metadata': 'False', 'n': 13, 'oh': 2048, 'ow': 1536, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.0105, 'crop': False, 'dh': 2048, 'dw': 1152, 'icc': True, 'metadata': 'False', 'n': 16, 'oh': 2048, 'ow': 1152, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.00396, 'crop': False, 'dh': 2048, 'dw': 1357, 'icc': True, 'metadata': 'False', 'n': 17, 'oh': 2048, 'ow': 1357, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'},
                                          {'a': 1.0, 'b': 0.0, 'bpp': 1.00498, 'crop': False, 'dh': 2048, 'dw': 1365, 'icc': True, 'metadata': 'False', 'n': 19, 'oh': 2048, 'ow': 1365, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.00708, 'crop': False, 'dh': 2048, 'dw': 1633, 'icc': True, 'metadata': 'False', 'n': 20, 'oh': 2048, 'ow': 1633, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}],
                          'le512/port': [{'a': 1.0, 'b': 0.0, 'bpp': 1.08312, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'False', 'n': 21, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.0857, 'crop': False, 'dh': 512, 'dw': 444, 'icc': True, 'metadata': 'False', 'n': 22, 'oh': 512, 'ow': 444, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.08535, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'False', 'n': 23, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.08301, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'False', 'n': 24, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'},
                                         {'a': 1.0, 'b': 0.0, 'bpp': 1.08566, 'crop': False, 'dh': 512, 'dw': 444, 'icc': True, 'metadata': 'False', 'n': 25, 'oh': 512, 'ow': 444, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.08547, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'False', 'n': 26, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.08302, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'False', 'n': 27, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.08574, 'crop': False, 'dh': 512, 'dw': 444, 'icc': True, 'metadata': 'False', 'n': 28, 'oh': 512, 'ow': 444, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'},
                                         {'a': 1.0, 'b': 0.0, 'bpp': 1.08546, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'False', 'n': 29, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}, {'a': 1.0, 'b': 0.0, 'bpp': 1.08303, 'crop': False, 'dh': 512, 'dw': 443, 'icc': True, 'metadata': 'False', 'n': 30, 'oh': 512, 'ow': 443, 'progressive': False, 'qf': 98.0, 'ss': 1.0, 'subsamp': '4:4:4', 'verdict': 'container_or_metadata_only'}]},
 'kakao|video|single': {'1080p/land': [{'a': 0.999916, 'b': -8e-06, 'bframes': True, 'bitrate_ratio': 0.41141, 'codec': 'hevc', 'crop': False, 'dh': 1080, 'dw': 1920, 'effective_crf': 20.87, 'fps': 30.0, 'gop': 28.0, 'n': 1, 'oh': 1080, 'ow': 1920, 'ss': 0.986259, 'temporal_offset': 1.0, 'verdict': 'transcoded_different_format', 'vprofile': 'Main'}, {'a': 0.999378, 'b': 0.00019, 'bframes': True, 'bitrate_ratio': 0.28322, 'codec': 'hevc', 'crop': False, 'dh': 1080, 'dw': 1920, 'effective_crf': 23.77, 'fps': 30.0, 'gop': 28.0, 'n': 2, 'oh': 1080, 'ow': 1920, 'ss': 0.982454, 'temporal_offset': 1.0, 'verdict': 'transcoded_different_format', 'vprofile': 'Main'}, {'a': 0.999995, 'b': 4.1e-05, 'bframes': True, 'bitrate_ratio': 0.30574, 'codec': 'hevc', 'crop': False, 'dh': 1080, 'dw': 1920, 'effective_crf': 21.66, 'fps': 30.0, 'gop': 28.0, 'n': 3, 'oh': 1080, 'ow': 1920, 'ss': 0.98893, 'temporal_offset': 1.0, 'verdict': 'transcoded_different_format', 'vprofile': 'Main'},
                                       {'a': 0.999948, 'b': 0.000109, 'bframes': True, 'bitrate_ratio': 0.24793, 'codec': 'hevc', 'crop': False, 'dh': 1080, 'dw': 1920, 'effective_crf': 21.0, 'fps': 30.0, 'gop': 28.0, 'n': 4, 'oh': 1080, 'ow': 1920, 'ss': 0.984557, 'temporal_offset': 1.0, 'verdict': 'transcoded_different_format', 'vprofile': 'Main'}, {'a': 0.99953, 'b': 0.000293, 'bframes': True, 'bitrate_ratio': 0.43322, 'codec': 'hevc', 'crop': False, 'dh': 1080, 'dw': 1920, 'effective_crf': 20.55, 'fps': 30.0, 'gop': 28.0, 'n': 5, 'oh': 1080, 'ow': 1920, 'ss': 0.98371, 'temporal_offset': 1.0, 'verdict': 'transcoded_different_format', 'vprofile': 'Main'}, {'a': 0.999612, 'b': 0.000221, 'bframes': True, 'bitrate_ratio': 0.36636, 'codec': 'hevc', 'crop': False, 'dh': 1080, 'dw': 1920, 'effective_crf': 26.0, 'fps': 30.0, 'gop': 28.0, 'n': 13, 'oh': 1080, 'ow': 1920, 'ss': 0.985743, 'temporal_offset': 1.0, 'verdict': 'transcoded_different_format', 'vprofile': 'Main'},
                                       {'a': 0.999786, 'b': 0.000164, 'bframes': True, 'bitrate_ratio': 0.70773, 'codec': 'hevc', 'crop': False, 'dh': 1080, 'dw': 1920, 'effective_crf': 19.0, 'fps': 30.0, 'gop': 28.0, 'n': 15, 'oh': 1080, 'ow': 1920, 'ss': 0.993563, 'temporal_offset': 1.0, 'verdict': 'transcoded_different_format', 'vprofile': 'Main'}],
                        '1080p/port': [{'a': 0.999766, 'b': 0.00019, 'bframes': True, 'bitrate_ratio': 0.30559, 'codec': 'hevc', 'crop': False, 'dh': 1920, 'dw': 1080, 'effective_crf': 27.0, 'fps': 30.0, 'gop': 28.0, 'n': 11, 'oh': 1920, 'ow': 1080, 'ss': 0.986318, 'temporal_offset': 1.0, 'verdict': 'transcoded_different_format', 'vprofile': 'Main'}, {'a': 0.998186, 'b': 0.000808, 'bframes': True, 'bitrate_ratio': 0.25554, 'codec': 'hevc', 'crop': False, 'dh': 1920, 'dw': 1080, 'effective_crf': 29.69, 'fps': 30.0, 'gop': 28.0, 'n': 12, 'oh': 1920, 'ow': 1080, 'ss': 0.966007, 'temporal_offset': 1.0, 'verdict': 'transcoded_different_format', 'vprofile': 'Main'}, {'a': 0.998577, 'b': 0.000388, 'bframes': True, 'bitrate_ratio': 0.24341, 'codec': 'hevc', 'crop': False, 'dh': 1920, 'dw': 1080, 'effective_crf': 29.0, 'fps': 30.0, 'gop': 28.0, 'n': 14, 'oh': 1920, 'ow': 1080, 'ss': 0.978375, 'temporal_offset': 1.0, 'verdict': 'transcoded_different_format', 'vprofile': 'Main'}],
                        '2160p/land': [{'a': 0.989946, 'b': 0.006093, 'bframes': True, 'bitrate_ratio': 0.05371, 'codec': 'hevc', 'crop': False, 'dh': 1080, 'dw': 1920, 'effective_crf': 36.25, 'fps': 30.0, 'gop': 28.0, 'n': 23, 'oh': 2160, 'ow': 3840, 'ss': 0.908578, 'temporal_offset': 1.0, 'verdict': 'transcoded_different_format', 'vprofile': 'Main'}, {'a': 1.001226, 'b': 0.000724, 'bframes': True, 'bitrate_ratio': 0.08385, 'codec': 'hevc', 'crop': False, 'dh': 1080, 'dw': 1920, 'effective_crf': 29.0, 'fps': 30.0, 'gop': 28.0, 'n': 24, 'oh': 2160, 'ow': 3840, 'ss': 0.961634, 'temporal_offset': 1.0, 'verdict': 'transcoded_different_format', 'vprofile': 'Main'}, {'a': 1.002763, 'b': 0.001001, 'bframes': True, 'bitrate_ratio': 0.17295, 'codec': 'hevc', 'crop': False, 'dh': 1080, 'dw': 1920, 'effective_crf': 19.33, 'fps': 30.0, 'gop': 28.0, 'n': 25, 'oh': 2160, 'ow': 3840, 'ss': 0.985419, 'temporal_offset': 1.0, 'verdict': 'transcoded_different_format', 'vprofile': 'Main'},
                                       {'a': 0.999061, 'b': 0.003082, 'bframes': True, 'bitrate_ratio': 0.21047, 'codec': 'hevc', 'crop': False, 'dh': 1080, 'dw': 1920, 'effective_crf': 18.42, 'fps': 30.0, 'gop': 28.0, 'n': 27, 'oh': 2160, 'ow': 3840, 'ss': 0.983867, 'temporal_offset': 1.0, 'verdict': 'transcoded_different_format', 'vprofile': 'Main'}],
                        '2160p/port': [{'a': 1.001657, 'b': 0.001167, 'bframes': True, 'bitrate_ratio': 0.09074, 'codec': 'hevc', 'crop': False, 'dh': 1920, 'dw': 1080, 'effective_crf': 25.54, 'fps': 30.0, 'gop': 28.0, 'n': 21, 'oh': 3840, 'ow': 2160, 'ss': 0.969339, 'temporal_offset': 1.0, 'verdict': 'transcoded_different_format', 'vprofile': 'Main'}, {'a': 0.998621, 'b': 0.003317, 'bframes': True, 'bitrate_ratio': 0.09907, 'codec': 'hevc', 'crop': False, 'dh': 1920, 'dw': 1080, 'effective_crf': 24.31, 'fps': 30.0, 'gop': 28.0, 'n': 22, 'oh': 3840, 'ow': 2160, 'ss': 0.976425, 'temporal_offset': 1.0, 'verdict': 'transcoded_different_format', 'vprofile': 'Main'}, {'a': 0.999415, 'b': 0.003032, 'bframes': True, 'bitrate_ratio': 0.09767, 'codec': 'hevc', 'crop': False, 'dh': 1920, 'dw': 1064, 'effective_crf': 26.19, 'fps': 30.0, 'gop': 28.0, 'n': 26, 'oh': 3840, 'ow': 2132, 'ss': 0.966139, 'temporal_offset': 1.0, 'verdict': 'transcoded_different_format', 'vprofile': 'Main'},
                                       {'a': 0.998246, 'b': 0.003477, 'bframes': True, 'bitrate_ratio': 0.08962, 'codec': 'hevc', 'crop': False, 'dh': 1920, 'dw': 1080, 'effective_crf': 28.53, 'fps': 30.0, 'gop': 28.0, 'n': 28, 'oh': 3840, 'ow': 2160, 'ss': 0.966212, 'temporal_offset': 1.0, 'verdict': 'transcoded_different_format', 'vprofile': 'Main'}, {'a': 0.996969, 'b': 0.00413, 'bframes': True, 'bitrate_ratio': 0.10673, 'codec': 'hevc', 'crop': False, 'dh': 1920, 'dw': 1080, 'effective_crf': 25.64, 'fps': 30.0, 'gop': 28.0, 'n': 29, 'oh': 3840, 'ow': 2160, 'ss': 0.970969, 'temporal_offset': 1.0, 'verdict': 'transcoded_different_format', 'vprofile': 'Main'}, {'a': 0.999196, 'b': 0.003037, 'bframes': True, 'bitrate_ratio': 0.10903, 'codec': 'hevc', 'crop': False, 'dh': 1920, 'dw': 1080, 'effective_crf': 26.61, 'fps': 30.0, 'gop': 28.0, 'n': 30, 'oh': 3840, 'ow': 2160, 'ss': 0.977733, 'temporal_offset': 1.0, 'verdict': 'transcoded_different_format', 'vprofile': 'Main'}],
                        '720p/land': [{'a': 1.015213, 'b': -0.00941, 'bframes': False, 'bitrate_ratio': 0.37548, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 23.44, 'fps': 24.0, 'gop': 24.0, 'n': 6, 'oh': 720, 'ow': 1280, 'ss': 0.933425, 'temporal_offset': 3.0, 'verdict': 'recompressed_same_geometry', 'vprofile': 'Constrained Baseline'}, {'a': 0.989268, 'b': 0.000504, 'bframes': False, 'bitrate_ratio': 0.35718, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 24.0, 'fps': 24.0, 'gop': 24.0, 'n': 7, 'oh': 720, 'ow': 1280, 'ss': 0.91856, 'temporal_offset': 4.0, 'verdict': 'recompressed_same_geometry', 'vprofile': 'Constrained Baseline'}, {'a': 0.996904, 'b': 0.003907, 'bframes': False, 'bitrate_ratio': 0.39432, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 23.0, 'fps': 24.0, 'gop': 24.0, 'n': 8, 'oh': 720, 'ow': 1280, 'ss': 0.928163, 'temporal_offset': 3.0, 'verdict': 'recompressed_same_geometry', 'vprofile': 'Constrained Baseline'},
                                      {'a': 0.986023, 'b': -5.4e-05, 'bframes': False, 'bitrate_ratio': 0.26518, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 25.6, 'fps': 24.0, 'gop': 22.5, 'n': 9, 'oh': 720, 'ow': 1280, 'ss': 0.886603, 'temporal_offset': 5.0, 'verdict': 'recompressed_same_geometry', 'vprofile': 'Constrained Baseline'}, {'a': 1.005762, 'b': -0.004295, 'bframes': False, 'bitrate_ratio': 0.48737, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 21.39, 'fps': 24.0, 'gop': 24.0, 'n': 10, 'oh': 720, 'ow': 1280, 'ss': 0.9138, 'temporal_offset': 5.0, 'verdict': 'recompressed_same_geometry', 'vprofile': 'Constrained Baseline'}, {'a': 0.961022, 'b': 0.017141, 'bframes': False, 'bitrate_ratio': 0.30343, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 27.79, 'fps': 24.0, 'gop': 20.636, 'n': 16, 'oh': 720, 'ow': 1280, 'ss': 0.812504, 'temporal_offset': 4.0, 'verdict': 'recompressed_same_geometry', 'vprofile': 'Constrained Baseline'},
                                      {'a': 0.986719, 'b': 0.012655, 'bframes': False, 'bitrate_ratio': 0.25453, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 1280, 'effective_crf': 30.0, 'fps': 24.0, 'gop': 18.25, 'n': 19, 'oh': 720, 'ow': 1280, 'ss': 0.788065, 'temporal_offset': 5.0, 'verdict': 'recompressed_same_geometry', 'vprofile': 'Constrained Baseline'}],
                        '720p/port': [{'a': 1.028774, 'b': -0.026805, 'bframes': False, 'bitrate_ratio': 0.76684, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 406, 'effective_crf': 18.0, 'fps': 24.0, 'gop': 24.0, 'n': 17, 'oh': 1280, 'ow': 720, 'ss': 0.974881, 'temporal_offset': 5.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Constrained Baseline'}, {'a': 0.991574, 'b': 0.006375, 'bframes': False, 'bitrate_ratio': 0.28631, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 406, 'effective_crf': 19.36, 'fps': 24.0, 'gop': 20.0, 'n': 18, 'oh': 1280, 'ow': 720, 'ss': 0.83848, 'temporal_offset': 4.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Constrained Baseline'}, {'a': 0.995373, 'b': 0.00486, 'bframes': False, 'bitrate_ratio': 0.21964, 'codec': 'h264', 'crop': False, 'dh': 720, 'dw': 406, 'effective_crf': 22.0, 'fps': 24.0, 'gop': 19.833, 'n': 20, 'oh': 1280, 'ow': 720, 'ss': 0.834822, 'temporal_offset': 1.0, 'verdict': 'resized_and_recompressed', 'vprofile': 'Constrained Baseline'}]}}

PLATFORMS = ('clean','facebook','instagram','kakao')
IMAGE_SUFFIXES = {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}
VIDEO_SUFFIXES = {'.mp4','.mov','.mkv','.avi','.webm'}

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
    """Frozen joint empirical observations: no CSV IO at runtime.

    Each row preserves geometry, codec/QF, temporal settings and photometry as
    observed together. Do not independently sample each parameter.
    Profiles are conditionally sampled for platform/quality/band/orientation and
    nearby aspect ratios; 360 unique observations (Instagram three folders
    collapsed to one profile per modality). This is not a population frequency.
    """
    def __init__(self):
        self.data = EMPIRICAL_POLICY

    def show(self):
        print('PROFILE | BAND/ORIENT | N | OBSERVED VERDICTS')
        for key, groups in sorted(self.data.items()):
            for category, variants in sorted(groups.items()):
                verdicts = Counter(v['verdict'] for v in variants)
                print(f'{key:26} {category:14} {len(variants):2d} {dict(verdicts)}')

    def choose(self, platform: str, modality: str, quality: str,
               band: str, width: int, height: int, rng: random.Random):
        key = f'{platform}|{modality}|{quality}'
        orientation = 'land' if width > height else 'port' if height > width else 'square'
        category = f'{band}/{orientation}'
        observations = self.data.get(key, {}).get(category)
        if not observations:
            raise ValueError(f'No measured profile for {key} at {category}; no invented OOD policy')
        # Log-aspect tolerance is a DESIGN assumption, not measured: only select
        # similarly shaped empirical inputs to avoid mixing panoramas and portraits.
        query_aspect = math.log(width / height)
        distances = [abs(query_aspect - math.log(x['ow']/x['oh'])) for x in observations]
        best = min(distances)
        if best > 0.15:
            raise ValueError(f'Input aspect ratio {width}/{height} too far from observed '
                             f'{key}/{category} (distance {best:.3f}>0.15); refuse extrapolation')
        shortlist = [v for v,d in zip(observations,distances) if d <= max(0.025,best+0.025)]
        v = rng.choice(shortlist)
        # Actual observed row is chosen as a correlated tuple; no aggregation.
        row = {
            'stimulus_id': f'embedded:{key}:{category}:row_{v["n"]:02d}',
            'target_id': key,
            'reference_profile_row_ordinal': v['n'],
            'reencode_verdict': v['verdict'],
            'orig_width': str(v['ow']), 'orig_height': str(v['oh']),
            'down_width': str(v['dw']), 'down_height': str(v['dh']),
            'crop_applied': str(v['crop']),
            'down_estimated_jpeg_qf': str(v.get('qf') or ''),
            'down_chroma_subsampling': v.get('subsamp','4:2:0'),
            'down_jpeg_progressive': str(v.get('progressive',False)),
            'down_has_icc': str(v.get('icc',False)),
            'metadata_stripped': str(v.get('metadata','')),
            'brightness_offset_b': str(v.get('b') if v.get('b') is not None else 0),
            'contrast_gain_a': str(v.get('a') if v.get('a') is not None else 1),
            'displayed_brightness_offset_b_median': str(v.get('b') if v.get('b') is not None else 0),
            'displayed_contrast_gain_a_median': str(v.get('a') if v.get('a') is not None else 1),
            'down_codec_name': v.get('codec','h264'),
            'down_profile': v.get('vprofile',''),
            'down_avg_frame_rate': str(v.get('fps') or ''),
            'down_gop_mean': str(v.get('gop') or 150),
            'down_has_b_frames': str(v.get('bframes',False)),
            'bitrate_ratio': str(v.get('bitrate_ratio') or ''),
            'effective_crf_interp': str(v.get('effective_crf') or ''),
            'temporal_offset_frames': str(v.get('temporal_offset') or 0),
            'reference_bpp_ratio': v.get('bpp'), 'reference_ssim': v.get('ss'),
        }
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


def fit_luma_ab(reference, rendered) -> dict[str, float]:
    """Estimate output Y ~= a * input Y + b, on registered same-size frames.

    Y is normalized [0,1] after Pillow/FFmpeg 8-bit conversion. This is a
    deliberately documented approximation to the research photometric fitter.
    """
    import numpy as np
    from PIL import Image
    if reference.size != rendered.size:
        raise ValueError('Photometric calibration needs equal image geometry')
    x = np.asarray(reference.convert('YCbCr').getchannel('Y'), dtype=np.float64).ravel()/255.0
    y = np.asarray(rendered.convert('YCbCr').getchannel('Y'), dtype=np.float64).ravel()/255.0
    step = max(1, int(math.ceil(x.size/150_000)))
    x = x[::step]; y = y[::step]
    # Avoid fitting saturated pixels when the transform clips them.
    valid = (x > 0.015) & (x < 0.985) & (y > 0.005) & (y < 0.995)
    if valid.sum() < 64: valid = np.ones_like(x,dtype=bool)
    x=x[valid]; y=y[valid]
    variance=float(np.var(x))
    if variance < 1e-8:
        return {'a':1.0,'b':float(np.mean(y-x)),'n':int(x.size),'low_variance':True}
    slope=float(np.mean((x-x.mean())*(y-y.mean()))/variance)
    intercept=float(y.mean()-slope*x.mean())
    return {'a':slope,'b':intercept,'n':int(x.size),'low_variance':False}


def photo_residual(fit: dict[str,float], target: dict[str,float]) -> float:
    # Balance intercept and contrast relative to their declared tolerances.
    return abs(fit['b']-target['b'])/0.004 + abs(fit['a']-target['a'])/0.01


def _video_y_frames(path: Path, filters: list[str], n: int) -> list:
    """Four aligned small reference/output frames, avoiding full decode in RAM."""
    import numpy as np
    filt=filters+['scale=160:160:flags=bilinear','format=gray']
    cmd=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-i',str(path),
         '-map','0:v:0','-an','-vf',','.join(filt),'-frames:v',str(n),
         '-f','rawvideo','-pix_fmt','gray','pipe:1']
    proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if proc.returncode:
        raise RuntimeError('photometric frame decode failed: '+proc.stderr.decode(errors='replace')[-800:])
    size=160*160
    arr=np.frombuffer(proc.stdout,dtype=np.uint8)
    if arr.size < size:
        raise RuntimeError('No frames available for video photometric calibration')
    frame_count=arr.size//size
    return arr[:frame_count*size].reshape(frame_count,160,160)


def fit_video_photo(input_src: Path, encoded: Path, preencode_filters: list[str], nframes: int) -> dict:
    """Compare geometrically/temporally aligned decoded luma from two videos.

    This is a low-cost approximate fit; do NOT label it as the original CSV's
    detailed displayed-Y photometric estimator. Camera/video range differs.
    """
    import numpy as np
    x=_video_y_frames(input_src,preencode_filters,nframes)
    y=_video_y_frames(encoded,[],nframes)
    usable=min(len(x),len(y));x=x[:usable].astype(np.float64).ravel()/255.0
    y=y[:usable].astype(np.float64).ravel()/255.0
    valid=(x>0.015)&(x<0.985)&(y>0.005)&(y<0.995)
    if valid.sum()<64:valid=np.ones_like(x,dtype=bool)
    x=x[valid];y=y[valid]
    v=float(np.var(x))
    a=float(np.mean((x-x.mean())*(y-y.mean()))/v) if v>1e-8 else 1.0
    b=float(y.mean()-a*x.mean())
    return {'a':a,'b':b,'n_frames':usable,'n_pixels':int(x.size),
            'warning':'8-bit gray FFmpeg sampling, not CSV displayed-Y estimator'}

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
        shutil.copy2(src,dst)
        if mask_src and mask_dst:
            mask_dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(mask_src,mask_dst)
        return {'verdict':verdict,'pixel_transform':'identity','photo':{'b':0,'a':1,
                'applied':False},'jpeg':None,'geometry':'preserved',
                'note':'Pixel identity approximation; metadata removal not reproduced.'}

    with Image.open(src) as opened:
        im = ImageOps.exif_transpose(opened).convert('RGB')
        orig_icc=opened.info.get('icc_profile')
    crop_rect,target,geo=image_geometry(im.width,im.height,row,platform,rng,args)
    if crop_rect is not None: im=im.crop(crop_rect)
    if im.size!=target: im=im.resize(target,Image.Resampling.LANCZOS)
    geometric_reference=im.copy()
    photo=photo_targets(row,'image',rng,args)
    qf=inum(row.get('down_estimated_jpeg_qf'),95)
    qf=int(round(clamp(qf+rng.uniform(-args.jpeg_qf_jitter,args.jpeg_qf_jitter),1,100)))
    subsamp=row.get('down_chroma_subsampling','4:2:0')
    subsampling={'4:4:4':0,'4:2:2':1,'4:2:0':2}.get(subsamp,2)
    progressive=flag(row.get('down_jpeg_progressive'))
    kwargs={'format':'JPEG','quality':qf,'subsampling':subsampling,'progressive':progressive}
    if flag(row.get('down_has_icc')) and orig_icc:kwargs['icc_profile']=orig_icc

    def encode_and_fit(image):
        buffer=io.BytesIO();image.save(buffer,**kwargs)
        payload=buffer.getvalue()
        with Image.open(io.BytesIO(payload)) as decoded:
            fit=fit_luma_ab(geometric_reference,decoded.convert('RGB'))
        return payload,fit

    baseline_blob,baseline_fit=encode_and_fit(im)
    final_blob,final_fit=baseline_blob,baseline_fit
    correction={'a':1.0,'b':0.0,'clamped':False}
    if photo['applied'] and not baseline_fit.get('low_variance'):
        a0=baseline_fit['a']
        if abs(a0)>0.05:
            raw_a=photo['a']/a0
            raw_b=(photo['b']-baseline_fit['b'])/a0
            correction['a']=clamp(raw_a,0.94,1.06)
            correction['b']=clamp(raw_b,-0.04,0.04)
            correction['clamped']=(correction['a']!=raw_a or correction['b']!=raw_b)
            if abs(correction['a']-1)>1e-7 or abs(correction['b'])>1e-7:
                trial_blob,trial_fit=encode_and_fit(adjust_luma_pillow(im,correction['a'],correction['b']))
                if photo_residual(trial_fit,photo)<photo_residual(baseline_fit,photo):
                    final_blob,final_fit=trial_blob,trial_fit
                else:
                    correction['rejected_if_worse']=True
    dst.write_bytes(final_blob)
    if mask_src and mask_dst:
        with Image.open(mask_src) as opened: mask=opened.copy()
        if crop_rect is not None:mask=mask.crop(crop_rect)
        if mask.size!=target:mask=mask.resize(target,Image.Resampling.NEAREST)
        mask_dst.parent.mkdir(parents=True,exist_ok=True);mask.save(mask_dst)
    return {'verdict':verdict,'pixel_transform':'resized_or_recompressed',
            'geometry':geo,'photo':dict(photo,baseline_estimate=baseline_fit,
                    final_estimate=final_fit,preencode_compensation=correction,
                    estimator='8-bit YCbCr OLS after exact synthetic geometry; NOT original CSV estimator'),
            'jpeg':{'qf':qf,'subsampling':subsamp,'progressive':progressive,
                    'metadata_preserved':False},
            'reference':{'bpp_ratio':row.get('reference_bpp_ratio'),
                         'ssim':row.get('reference_ssim')}}


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
    # No pre-encoding photometric transform yet; measure codec-only outcome.
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

    def encode(destination: Path, extra_photo: Optional[tuple[float,float]]):
        vf=filters.copy()
        if extra_photo is not None:
            ga,gb=extra_photo
            if abs(ga-1)>1e-7 or abs(gb)>1e-7:
                # Apply an affine transform directly to coded luma (219-level TV scale).
                vf.append(f'lutyuv=y=val*{ga:.7f}+{gb*219:.7f}')
        cmd=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y',
             '-i',str(src),'-map','0:v:0','-map','0:a?']
        if vf:cmd+=['-vf',','.join(vf)]
        cmd+=codec_args+['-r',f'{out_fps:.8g}','-fps_mode','cfr',
                          '-color_range','tv','-c:a','aac','-b:a','128k',
                          '-movflags','+faststart',str(destination)]
        try:run(cmd)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError('ffmpeg encode failed: '+exc.stderr[-1600:]) from exc

    photo_info=dict(photo,estimator='FFmpeg decoded gray OLS, NOT original displayed-Y estimator')
    if photo['applied']:
        # Two-stage calibration: first encode WITHOUT extra brightness/contrast,
        # measure its codec effect, then apply only the residual target.
        # This avoids double-counting b/a already produced by the codec.
        with tempfile.TemporaryDirectory(prefix='calibrate_',dir=dst.parent) as temp_dir:
            temp=Path(temp_dir)/'baseline.mp4'
            encode(temp,None)
            baseline=fit_video_photo(src,temp,filters,args.photometric_sample_frames)
            photo_info['codec_only_estimate']=baseline
            chosen=temp
            raw_a=photo['a']/baseline['a'] if abs(baseline['a'])>0.05 else 1.0
            raw_b=(photo['b']-baseline['b'])/baseline['a'] if abs(baseline['a'])>0.05 else 0.0
            ga=clamp(raw_a,0.94,1.06);gb=clamp(raw_b,-0.04,0.04)
            photo_info['preencode_compensation']={'a':ga,'b':gb,
                'clamped':ga!=raw_a or gb!=raw_b}
            if abs(ga-1)>1e-7 or abs(gb)>1e-7:
                trial=Path(temp_dir)/'compensated.mp4'
                encode(trial,(ga,gb))
                trial_fit=fit_video_photo(src,trial,filters,args.photometric_sample_frames)
                photo_info['compensated_estimate']=trial_fit
                if photo_residual(trial_fit,photo)<photo_residual(baseline,photo):
                    chosen=trial
                else:
                    photo_info['compensation_rejected_if_worse']=True
                # Optional second residual correction, bounded and accepted only
                # if the actual decoded output moves closer to the observed a/b.
                if photo_residual(trial_fit,photo)>0.15:
                    da=ga-1.0
                    local_slope=(trial_fit['a']-baseline['a'])/da if abs(da)>1e-4 else 1.0
                    if abs(local_slope)<0.25:local_slope=1.0
                    gain2=clamp(ga+(photo['a']-trial_fit['a'])/local_slope,0.94,1.06)
                    offset2=clamp(gb+(photo['b']-trial_fit['b'])*255/219,-0.04,0.04)
                    if abs(gain2-ga)>1e-6 or abs(offset2-gb)>1e-6:
                        trial2=Path(temp_dir)/'compensated_2.mp4'
                        encode(trial2,(gain2,offset2))
                        trial2_fit=fit_video_photo(src,trial2,filters,args.photometric_sample_frames)
                        photo_info['second_compensation']={'a':gain2,'b':offset2,
                                                              'estimate':trial2_fit}
                        if photo_residual(trial2_fit,photo)<photo_residual(
                                fit_video_photo(src,chosen,filters,args.photometric_sample_frames),photo):
                            chosen=trial2
            shutil.copy2(chosen,dst)
            photo_info['final_estimate']=fit_video_photo(src,dst,filters,args.photometric_sample_frames)
            photo_info['residual_in_estimator_units']={
                'b':round(photo_info['final_estimate']['b']-photo['b'],7),
                'a':round(photo_info['final_estimate']['a']-photo['a'],7)}
    else:
        encode(dst,None)

    out_probe=ffprobe_json(dst) if not args.skip_video_output_probe else None
    return {'verdict':row['reencode_verdict'],'geometry':geometry,
            'photo':photo_info,'video':{
                'codec':codec,'profile':profile,'fps':out_fps,'input_fps':in_fps,
                'reference_gop':gop,'bframes':bframes,
                'h264_crf':None if is_hevc else round(crf,3),
                'hevc_target_bitrate_bps':bitrate_target_bps,
                'reference_bitrate_ratio':fnum(row.get('bitrate_ratio')),
                'reference_ssim':row.get('reference_ssim'),
                'input_bitrate_bps':probe['bitrate_bps'],
                'output_probe':out_probe,
                'bitrate_match_warning':(
                    'Target bitrate ratio not reached; recalibrate on full clips.'
                    if is_hevc and out_probe and probe['bitrate_bps'] and
                    out_probe['bitrate_bps'] and bitrate_target_bps and
                    abs(out_probe['bitrate_bps']-bitrate_target_bps)/bitrate_target_bps>0.30
                    else None),
                'actual_bitrate_ratio':(
                    round(out_probe['bitrate_bps']/probe['bitrate_bps'],5)
                    if out_probe and probe['bitrate_bps'] and out_probe['bitrate_bps']
                    else None),
                'temporal_offset_frames_used':drop_frames,
                'ffmpeg_filter':','.join(filters),
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
    ap.add_argument('--p-photo', type=float, default=1.0,
                    help='Apply measured small brightness/contrast target on reencoded items')
    ap.add_argument('--photometric-sample-frames',type=int,default=4,
                    help='Sample first N aligned frames to estimate existing codec photometric change')
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
    if not (1 <= a.photometric_sample_frames <= 30):
        raise ValueError('--photometric-sample-frames must be between 1 and 30')
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
                                       'reference_measurements_csv': 'embedded_empirical_joint_observation_no_runtime_csv',
                                       'reference_row_ordinal':row.get('reference_profile_row_ordinal')})
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
