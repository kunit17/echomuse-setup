import unittest
from stream_core import *
class StreamTests(unittest.TestCase):
    def test_sentences_preserve_all_words(self):
        text="How was your day today Tony? I've been thinking about you a lot lately. I had the strangest dream of you."
        parts=split_sentences(text)
        self.assertEqual(len(parts),3);self.assertEqual(' '.join(parts),text)
    def test_titles_and_decimal_preserved(self):
        self.assertEqual(split_sentences('Dr. Jones paid 3.14 dollars. Really?'),['Dr. Jones paid 3.14 dollars.','Really?'])
    def test_long_text_preserves_words(self):
        text=('long ' * 110).strip()+'.'
        self.assertEqual(' '.join(split_sentences(text)),text)
    def test_zero_codes_are_valid(self):validate_codes([128266+i*4096 for i in range(7)])
    def test_bad_position_and_incomplete_rejected(self):
        for v in [[128266],[128266]*7,[128266+i*4096+4096 for i in range(7)]]:
            with self.assertRaises(ValueError):validate_codes(v)
    def test_lookahead_and_final_flush(self):
        self.assertEqual(safe_end(12*2048,12),4*2048)
        self.assertEqual(safe_end(12*2048,12,True),12*2048)
        self.assertEqual(safe_end(4*2048,4),0)
    def test_underflow_measured(self):
        v=simulate([(1,.5),(2,.5)],.5)
        self.assertEqual(v['underruns'],1);self.assertAlmostEqual(v['underrun_seconds'],.5)
    def test_buffer_hides_jitter(self):
        self.assertEqual(simulate([(1,.3),(1.1,.3),(1.2,.3)],.5)['underruns'],0)
    def test_short_audio_flushes(self):self.assertEqual(simulate([(2,.1)],.5)['first_playback'],2)
if __name__=='__main__':unittest.main()
