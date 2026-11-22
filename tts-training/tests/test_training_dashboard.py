import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

base=Path(__file__).resolve().parents[1]
source=base/'server.py' if (base/'server.py').exists() else base/'scripts/training_dashboard.py'
spec=importlib.util.spec_from_file_location('dashboard',source)
dashboard=importlib.util.module_from_spec(spec);spec.loader.exec_module(dashboard)

class FeedTests(unittest.TestCase):
    def test_incomplete_metrics_append_does_not_break_feed(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'epochs.jsonl'
            row=dict(run_id='valid',stage='s1',epoch=1,validation={'nll_per_token':2})
            path.write_text(json.dumps(row)+'\n'+json.dumps({**row,'epoch':2})+'\n{"unfinished":')
            rows,warnings=dashboard.epoch_rows(path,'valid')
            self.assertEqual([r['epoch'] for r in rows],[1,2]);self.assertFalse(warnings)

    def test_foreign_run_and_nonfinite_metrics_are_not_displayed(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'epochs.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in [
                dict(run_id='other',stage='s1',epoch=1,validation={'loss':1}),
                dict(run_id='valid',stage='s1',epoch=2,validation={'loss':float('nan')})]))
            rows,_=dashboard.epoch_rows(path,'valid');self.assertFalse(rows)

    def test_patience_matches_meaningful_improvement(self):
        policy=dict(metric='loss',min_relative_improvement=.005)
        rows=[dict(epoch=i,validation={'loss':v}) for i,v in enumerate([2,1,.999,.998,.997])]
        self.assertEqual(dashboard.patience(rows,policy),3)

    def test_run_id_cannot_escape_report_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'reports').mkdir()
            (root/'reports/training-status.json').write_text(json.dumps({'run_id':'../../anything'}))
            self.assertFalse(dashboard.Feed(root).snapshot()['available'])

if __name__=='__main__':unittest.main()
