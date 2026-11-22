import sys
from pathlib import Path
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from training_run import should_stop,prune_checkpoints,POLICIES

class TrainingPolicyTests(unittest.TestCase):
    def rows(self,values,stage='s1'):
        return [dict(stage=stage,epoch=i+1,validation={POLICIES[stage]['metric']:v}) for i,v in enumerate(values)]

    def test_stops_after_five_epochs_without_improvement(self):
        self.assertFalse(should_stop(self.rows([1]*5),'s1'))
        self.assertTrue(should_stop(self.rows([1]*6),'s1'))
        self.assertFalse(should_stop(self.rows([1,.9,.8,.7,.6,.5]),'s1'))

    def test_sovits_runs_to_owner_minimum_even_after_plateau(self):
        self.assertFalse(should_stop(self.rows([1]*6,'s2'),'s2'))
        self.assertFalse(should_stop(self.rows([1]*14,'s2'),'s2'))
        self.assertTrue(should_stop(self.rows([1]*25,'s2'),'s2'))

    def test_sovits_cannot_stop_before_25(self):
        self.assertFalse(should_stop(self.rows([1]*24,'s2'),'s2'))
        self.assertTrue(should_stop(self.rows([1]*25,'s2'),'s2'))

    def test_explicit_epoch_targets(self):
        self.assertTrue(should_stop(self.rows([1/i for i in range(1,21)]),'s1'))
        self.assertTrue(should_stop(self.rows([1/i for i in range(1,26)],'s2'),'s2'))

    def test_keeps_best_and_latest_and_resume_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);exports=root/'checkpoints/gpt';full=root/'features/gpt-training/ckpt'
            exports.mkdir(parents=True);full.mkdir(parents=True);reports=root/'reports';reports.mkdir()
            rows=self.rows([3,1,2])
            for row in rows:
                i=row['epoch'];checkpoint=exports/f'e{i}.ckpt';resume=full/f'e{i}.ckpt'
                checkpoint.write_bytes(b'export');resume.write_bytes(b'resume')
                row.update(checkpoint=str(checkpoint),full_checkpoint=str(resume))
            run=dict(checkpoints=str(root/'checkpoints'),features=str(root/'features'),reports=str(reports))
            prune_checkpoints(run,rows,'s1')
            self.assertEqual({p.name for p in exports.iterdir()},{'e2.ckpt','e3.ckpt'})
            self.assertEqual({p.name for p in full.iterdir()},{'e2.ckpt','e3.ckpt'})

    def test_wont_prune_if_best_checkpoint_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=self.rows([1]);rows[0]['checkpoint']=str(root/'missing.ckpt')
            with self.assertRaises(RuntimeError):
                prune_checkpoints(dict(checkpoints=str(root/'checkpoints'),features=str(root/'features'),reports=str(root)),rows,'s1')

if __name__=='__main__':unittest.main()
