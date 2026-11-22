import sys,unittest,wave
from pathlib import Path
import numpy as np
SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
if not SCRIPTS.is_dir():SCRIPTS=Path(__file__).resolve().parent
sys.path.insert(0,str(SCRIPTS))
from audio_postprocess import clean_generated_tail

class TailTests(unittest.TestCase):
    def setUp(self):self.sr=24000
    def test_three_reported_bursts_are_removed_without_changing_speech_or_duration(self):
        with np.load(Path(__file__).with_name('fixtures')/'csm-tail-bursts.npz') as fixtures:
            for name in fixtures.files:
                with self.subTest(sample=name):
                    generated=fixtures[name]
                    cleaned,decision=clean_generated_tail(generated,self.sr)
                    self.assertTrue(decision['applied'],decision)
                    self.assertEqual(len(cleaned),len(generated))
                    np.testing.assert_array_equal(cleaned[:-2400],generated[:-2400])
                    self.assertTrue(np.all(cleaned[-2400:]==0))
    def test_continuous_speech_ending_is_unchanged(self):
        x=(.08*np.sin(2*np.pi*180*np.arange(self.sr)/self.sr)).astype('float32')
        y,d=clean_generated_tail(x,self.sr)
        self.assertFalse(d['applied']);np.testing.assert_array_equal(x,y)
    def test_voiced_short_final_word_after_pause_is_preserved(self):
        x=np.zeros(self.sr,dtype='float32')
        x[-2400:]=.05*np.sin(2*np.pi*180*np.arange(2400)/self.sr)
        y,d=clean_generated_tail(x,self.sr)
        self.assertFalse(d['applied']);np.testing.assert_array_equal(x,y)
    def test_unvoiced_high_frequency_ending_is_preserved(self):
        x=np.zeros(self.sr,dtype='float32')
        random=np.random.default_rng(31).normal(0,.015,2400)
        x[-2400:]=random-np.roll(random,1)
        y,d=clean_generated_tail(x,self.sr)
        self.assertFalse(d['applied']);np.testing.assert_array_equal(x,y)
    def test_already_quiet_end_is_unchanged(self):
        x=np.zeros(self.sr,dtype='float32');x[:12000]=.05
        y,d=clean_generated_tail(x,self.sr)
        self.assertFalse(d['applied']);np.testing.assert_array_equal(x,y)
    def test_loud_uncertain_end_is_preserved(self):
        x=np.zeros(self.sr,dtype='float32');x[-2400:]=.2
        y,d=clean_generated_tail(x,self.sr)
        self.assertFalse(d['applied']);np.testing.assert_array_equal(x,y)
    def test_short_audio_is_unchanged(self):
        x=np.ones(200,dtype='float32')*.01
        y,d=clean_generated_tail(x,self.sr)
        self.assertFalse(d['applied']);np.testing.assert_array_equal(x,y)
if __name__=='__main__':unittest.main()
