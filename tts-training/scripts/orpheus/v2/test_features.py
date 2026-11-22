from common import *
import unittest
class FeatureTests(unittest.TestCase):
    def test_interleave_roundtrip_preserves_repeated_frames(self):
        levels=[[7,7],[2,3,2,3],[4,5,6,7,4,5,6,7]]
        tokens=pack_codes(levels)
        self.assertEqual(len(tokens),14)
        self.assertEqual(unpack_codes(tokens),levels)
    def test_incomplete_group_rejected(self):
        with self.assertRaises(ValueError):unpack_codes([TOKENS['audio']]*6)
    def test_wrong_codebook_offset_rejected(self):
        with self.assertRaises(ValueError):unpack_codes([TOKENS['audio']]*7)
    def test_audio_endpoint_is_retained(self):
        codes=pack_codes([[9],[8,7],[6,5,4,3]])
        ids=format_example([128000,123],codes)
        self.assertEqual(ids[-2:],[TOKENS['end_speech'],TOKENS['end_ai']])
        self.assertEqual(ids[ids.index(TOKENS['start_speech'])+1:-2],codes)
    def test_padding_labels_ignored_without_truncation(self):
        examples=[dict(input_ids=list(range(n)),attention_mask=[1]*n,labels=list(range(n))) for n in [2499,17]]
        batch=collate(examples)
        self.assertEqual(batch['input_ids'].shape,(2,2504))
        self.assertEqual(batch['labels'][0,:2499].tolist(),examples[0]['labels'])
        self.assertTrue((batch['labels'][0,2499:]==-100).all())
        self.assertTrue((batch['attention_mask'][1,17:]==0).all())
if __name__=='__main__':unittest.main()
