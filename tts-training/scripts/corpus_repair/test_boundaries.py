import importlib.util
from pathlib import Path
import numpy as np
import copy,json,tempfile

spec=importlib.util.spec_from_file_location('repair',Path(__file__).with_name('build.py'))
repair=importlib.util.module_from_spec(spec);spec.loader.exec_module(repair)

def test():
    sr=32000;x=np.zeros(sr,dtype='float32');x[8000:24000]=.1
    check=repair.edge_check(x,.003)
    assert check['leading_quiet'] and check['trailing_quiet']
    # A recognizable word can end in active audio: timestamps/text cannot pass it.
    unsafe=x.copy();unsafe[-640:]=.05
    assert not repair.edge_check(unsafe,.003)['trailing_quiet']
    unsafe=x.copy();unsafe[:640]=.05
    assert not repair.edge_check(unsafe,.003)['leading_quiet']
    # Fifty milliseconds of actual silence is insufficient for the 120 ms guard.
    unsafe=np.full(sr,.1,dtype='float32');unsafe[-1600:]=0
    assert not repair.edge_check(unsafe,.003)['trailing_quiet']
    assert repair.subtract([[0,30]],[[4,8],[12,15]])==[[0,4],[8,12],[15,30]]
    assert repair.merge([[3,5],[0,2],[1,4]])==[[0,5]]
    assert repair.runs(np.array([False]*10+[True]*30+[False]*10))==[(.1,.4)]
    assert repair.normalize_ctc('It was 30 years ago.')=='IT WAS THIRTY YEARS AGO'
    # Fresh-cache validation must not read deleted legacy manifests/Windows files.
    recipe=copy.deepcopy(repair.RECIPE);saved_work=repair.WORK
    with tempfile.TemporaryDirectory() as temporary:
        repair.WORK=Path(temporary)
        assert repair.validate_recipe(check_sources=False)
        # A held-out span must never also be assigned to training.
        source=next(iter(recipe['sources']))
        repair.RECIPE=copy.deepcopy(recipe)
        repair.RECIPE['partitions'][source]['train'].append(recipe['partitions'][source]['validation'][0])
        try:repair.validate_recipe(check_sources=False)
        except AssertionError:pass
        else:raise AssertionError('Overlapping training/validation source spans passed')
        repair.RECIPE=copy.deepcopy(recipe)
        repair.RECIPE['policy']['quiet_seconds']=.01
        try:repair.validate_recipe(check_sources=False)
        except AssertionError:pass
        else:raise AssertionError('Unsafe changed boundary policy passed')
        repair.RECIPE=copy.deepcopy(recipe)
        bad=dict(sources=recipe['sources'],partitions=recipe['partitions'],exclusions={s:[] for s in recipe['sources']},policy=recipe['policy'])
        (repair.WORK/'plan.json').write_text(json.dumps(bad))
        try:repair.validate_recipe(check_sources=False)
        except AssertionError:pass
        else:raise AssertionError('Cache with different exclusions passed')
    repair.RECIPE=recipe;repair.WORK=saved_work
    print('PASS: active-edge regression, insufficient quiet, exclusions, interval merging and number normalization')
    print('PASS: standalone recipe, held-out source overlap, unsafe policy and changed-cache exclusions')

if __name__=='__main__':test()
