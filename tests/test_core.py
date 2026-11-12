"""Small unit tests, not a substitute for the benchmark's numerical audits."""
from pathlib import Path
import importlib.util
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'benchmark'),str(ROOT/'scripts')]
from models import response,INITIAL,TIMES
from experiment import Bank,obs,dopt_scores,risk_scores
from prepare_visual_data import controls,stlsq

class CoreTests(unittest.TestCase):
    def test_exact_passive_ambiguity(self):
        self.assertTrue(np.allclose(response('chemostat',[.9,0],(1,)),response('chemostat',[0,.9],(1,))))
        self.assertFalse(np.allclose(response('chemostat',[.9,0],(.1,)),response('chemostat',[0,.9],(.1,))))
    def test_control_rank_and_recovery(self):
        for c in controls().values():
            self.assertEqual(c['rank_before'],1);self.assertEqual(c['rank_after'],2)
            np.testing.assert_allclose(c['new_coef'],c['true_coef'],atol=1e-10)
            np.testing.assert_allclose(c['old_initial'],c['true_initial'],atol=1e-10)
    def test_noise_is_reproducible(self):
        a=obs('enzyme',np.array([0.,.3]),4,.03,123,0)
        np.testing.assert_array_equal(a,obs('enzyme',np.array([0.,.3]),4,.03,123,0))
        self.assertFalse(np.array_equal(a,obs('enzyme',np.array([0.,.3]),4,.03,123,1)))
    def test_library_support_posterior(self):
        b=Bank('chemostat',24);y=response('chemostat',[0,.9],INITIAL['chemostat'])[0]
        history=[(-1,y)];w=b.posterior(history,.05);pm,_,_=b.summarize(w)
        self.assertAlmostEqual(w.sum(),1,places=10)
        self.assertAlmostEqual(pm[1],pm[2],places=10)
        d=dopt_scores(b,w,history,.05)
        self.assertTrue(np.isfinite(d).all());self.assertTrue((d>=-1e-10).all())
        risk,_=risk_scores(b,w,.05,power=3)
        self.assertTrue((risk>=0).all());self.assertTrue((risk<=1-pm.max()+1e-10).all())
    def test_joint_run_breaks_ambiguity(self):
        b=Bank('chemostat',40)
        y0=response('chemostat',[0,.9],(1,))[0];y1=response('chemostat',[0,.9],(.1,))[0]
        pm,which,_=b.summarize(b.posterior([(-1,y0),(0,y1)],.03))
        self.assertEqual(which,2);self.assertGreater(pm[which],.95)
    def test_fixed_acquisition_horizon(self):
        self.assertEqual(len(TIMES),16);self.assertEqual(float(TIMES[-1]),1.5)
        self.assertGreater(float(TIMES[0]),0)

if __name__=='__main__':unittest.main()
