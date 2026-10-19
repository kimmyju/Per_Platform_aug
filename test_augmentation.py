import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
import numpy as np
import augmentation_profiles as profiles
from online_augment import PlatformAwareOnlineImageAugmentor, RuntimeConfig

class PolicyTests(unittest.TestCase):
    def test_cardinality_and_fidelity(self):
        self.assertEqual(len(profiles.RECIPE_CATALOG), 12)
        self.assertEqual(sum(len(cells) for cells in profiles.RECIPE_CATALOG.values()),65)
        self.assertEqual(profiles.count_recipes(), 360)
        for cells in profiles.RECIPE_CATALOG.values():
            for group in cells.values():
                for r in group:
                    self.assertGreater(r['orig_width'], 0)
                    self.assertGreater(r['down_width'], 0)
                    self.assertIn(r['reencode_verdict'], {'container_or_metadata_only', 'recompressed_same_geometry','resized_and_recompressed','transcoded_different_format'})

    def test_conditional_mixture_and_seed(self):
        a=profiles.sample_policy('image',1080,608,key='pair1',base_seed=42,force_platform='facebook',force_quality='low')
        b=profiles.sample_policy('image',1080,608,key='pair1',base_seed=42,force_platform='facebook',force_quality='low')
        self.assertEqual(a['recipe']['id'], b['recipe']['id'])
        self.assertEqual(a['band'], 'le1080')
        self.assertEqual(a['orientation'], 'land')

    def test_image_real_compress_and_mask(self):
        x=np.random.default_rng(6).integers(20,240,size=(608,1080,3),dtype=np.uint8)
        im=Image.fromarray(x)
        mask=Image.new('L',(1080,608),1)
        aug=PlatformAwareOnlineImageAugmentor(RuntimeConfig(seed=42))
        out, m, info=aug(im,key='abc',mask=mask,force_platform='instagram',return_info=True)
        self.assertEqual(out.size[0],out.size[1])
        self.assertEqual(out.size,m.size)
        self.assertEqual(info['platform'],'instagram')

    def test_online_epoch(self):
        aug=PlatformAwareOnlineImageAugmentor(RuntimeConfig(seed=42))
        aug.set_epoch(2)
        self.assertEqual(aug.cfg.epoch,2)

if __name__=='__main__':
    unittest.main()