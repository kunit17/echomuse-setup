import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from corpus_review import structural_checks,quiet_metrics,repetition_flags,choose_queue

class ReviewTests(unittest.TestCase):
    def row(self,id,split,lo,hi,text='A complete sentence.',audio=None):
        return dict(id=id,split=split,source_start_frame=lo,source_end_frame=hi,text=text,
                    audio_sha256=audio or id,filename=id+'.wav',hard_errors=[],flags=[],
                    duration_seconds=5,speaker={'distance_to_training_center':.1},pitch_deviation=0)

    def test_leakage_and_containment(self):
        rows=[self.row('a','train',0,100),self.row('b','validation',90,150)]
        errors,duplicates=structural_checks(rows,{'train':[[0,80]],'validation':[[80,200]]})
        kinds={r['kind'] for r in errors}
        self.assertIn('outside_frozen_split',kinds);self.assertIn('cross_split_source_overlap',kinds)
        self.assertIn('text_duplicate',kinds)
        self.assertTrue(any(d['cross_split'] for d in duplicates))

    def test_pcm_duplicates_detected_with_different_filenames(self):
        rows=[self.row('a','train',0,50,'One sentence.',audio='same'),
              self.row('b','validation',100,150,'Different words.',audio='same')]
        errors,_=structural_checks(rows,{'train':[[0,80]],'validation':[[90,200]]})
        self.assertEqual([r['kind'] for r in errors],['audio_duplicate'])

    def test_quiet_boundaries_and_interior(self):
        audio=np.concatenate([np.zeros(3200),np.ones(3200),np.zeros(6400),np.ones(3200),np.zeros(4800)])
        result,_=quiet_metrics(audio,32000,.01)
        self.assertAlmostEqual(result['quiet_lead_seconds'],.1)
        self.assertAlmostEqual(result['quiet_tail_seconds'],.15)
        self.assertAlmostEqual(result['longest_interior_quiet_seconds'],.2)

    def test_repetition_without_rewriting(self):
        self.assertIn('repeated_word_or_phrase',repetition_flags('go home go home go home'))
        self.assertFalse(repetition_flags('Alex walked to the garden. Jordan stayed home.'))
        self.assertIn('repeated_sentence',repetition_flags('This is the full sentence. This is the full sentence.'))

    def test_deterministic_balanced_unique_queue(self):
        rows=[self.row(str(i),'train' if i<80 else 'validation',i*100,i*100+50,text=str(i)) for i in range(100)]
        queue=choose_queue(rows)
        self.assertEqual(len(queue),50);self.assertEqual(len({r['id'] for r in queue}),50)
        self.assertEqual(sum(r['split']=='validation' for r in queue),10)
        self.assertEqual([r['id'] for r in queue],[r['id'] for r in choose_queue(rows)])
        self.assertTrue(all('selection_reason' not in row for row in rows))

    def test_queue_covers_each_frozen_span(self):
        rows=[]
        for split,count in [('train',6),('validation',5)]:
            for block in range(count):
                for j in range(10):
                    row=self.row(f'{split}-{block}-{j}',split,block*10000+j*100,block*10000+j*100+50)
                    row['source_span']=block;rows.append(row)
        queue=choose_queue(rows)
        for split,count in [('train',6),('validation',5)]:
            self.assertEqual({r['source_span'] for r in queue if r['split']==split},set(range(count)))

if __name__=='__main__':unittest.main()
