"""Matched junction resampling uses real queries and leaves base training data intact."""
import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"impls"))
from utils.junction_sampling import JunctionRouteIndex,JunctionHighSampler,OFFSET_BIN_EDGES


def make_fixture(clean_count=5,previsit_goal=False,same_exit_count=0):
    W,J,E,S,G=(2,1),(2,2),(2,3),(3,2),(4,2)
    def episode(blocks):
        out=[]
        for cell,n in blocks:
            obs=np.zeros((n,29),np.float32)
            obs[:,:2]=np.array([cell[1],cell[0]])*4-4
            obs[:,3]=1
            out.append(obs)
        return np.concatenate(out)
    if previsit_goal:
        bad=[(W,10),(J,5),(E,20),(G,5),(E,35),(J,10),(S,20),(G,300)]
    else:
        bad=[(W,10),(J,5),(E,60),(J,10),(S,20),(G,300)]
    episodes=[episode(bad)]
    episodes += [episode([(W,10),(J,5),(S,20),(G,300)]) for _ in range(clean_count)]
    episodes += [episode([(W,10),(J,5),(E,20),(G,300)]) for _ in range(same_exit_count)]
    ends=np.cumsum([len(x) for x in episodes])-1
    return np.concatenate(episodes),ends


def query(batch=1,M=3):
    start=np.full(batch,12,np.int32)
    goal=np.full(batch,120,np.int32)
    targets=np.minimum(start[:,None]+50*np.arange(M,0,-1),goal[:,None]).astype(np.int32)
    return start,goal,targets


class JunctionSamplingTests(unittest.TestCase):
    def test_real_matched_queries_and_base_immutability(self):
        obs,ends=make_fixture()
        index=JunctionRouteIndex.build(obs,ends)
        sampler=JunctionHighSampler(index,obs,probability=1,seed=37)
        inputs=query(32);saved=tuple(x.copy() for x in inputs)
        result=sampler.sample(*inputs)
        self.assertTrue(result['replaced'].all())
        self.assertTrue(np.array_equal(result['support_visits'],np.full(32,5)))
        self.assertTrue(np.array_equal(result['support_episodes'],np.full(32,5)))
        for actual,old in zip(inputs,saved):np.testing.assert_array_equal(actual,old)
        s,g,t=result['start_indices'],result['goal_indices'],result['target_indices']
        np.testing.assert_array_equal(t,np.minimum(s[:,None]+50*np.arange(3,0,-1),g[:,None]))
        np.testing.assert_array_equal(index.frame_cells[s],index.frame_cells[inputs[0]])
        np.testing.assert_array_equal(index.frame_cells[g],index.frame_cells[inputs[1]])
        np.testing.assert_array_equal(np.searchsorted(OFFSET_BIN_EDGES,g-s,side='right'),np.searchsorted(OFFSET_BIN_EDGES,inputs[1]-inputs[0],side='right'))
        np.testing.assert_array_equal((g-s)[:,None]<[50,100,150],(inputs[1]-inputs[0])[:,None]<[50,100,150])
        self.assertTrue((index.frame_episodes[s]!=index.frame_episodes[inputs[0]]).all())
        donors=result['donor_visit_ids'];original=index.frame_visits[inputs[0]]
        self.assertTrue((index.events['first_exit'][donors]!=index.events['first_exit'][original]).all())
        self.assertTrue((t[:,-1]>=index.events['exit_index'][donors]).all())
        self.assertTrue(np.isfinite(list(result['metrics'].values())).all())

    def test_repeated_queries_do_not_create_independent_support(self):
        obs,ends=make_fixture(clean_count=4)
        sampler=JunctionHighSampler(JunctionRouteIndex.build(obs,ends),obs,probability=1)
        result=sampler.sample(*query(100))
        self.assertTrue(result['triggered'].all())
        self.assertFalse(result['supported'].any())
        self.assertFalse(result['replaced'].any())
        self.assertTrue((result['support_visits']==4).all())

    def test_same_exit_donors_cannot_fill_missing_alternative_support(self):
        obs,ends=make_fixture(clean_count=4,same_exit_count=10)
        sampler=JunctionHighSampler(JunctionRouteIndex.build(obs,ends),obs,probability=1)
        result=sampler.sample(*query(10))
        self.assertTrue((result['support_visits']==4).all())
        self.assertFalse(result['supported'].any())

    def test_same_episode_visits_cannot_fill_independent_alternatives(self):
        obs,ends=make_fixture(clean_count=6)
        # Join the recorded segments into one original episode. There are
        # multiple physical alternatives but no independent donor episode.
        index=JunctionRouteIndex.build(obs,np.array([len(obs)-1]))
        result=JunctionHighSampler(index,obs,probability=1).sample(*query(8))
        self.assertTrue(result['triggered'].all())
        self.assertTrue((result['support_visits']==0).all())
        self.assertFalse(result['replaced'].any())

    def test_goal_already_visited_before_return_is_not_trigger(self):
        obs,ends=make_fixture(previsit_goal=True)
        sampler=JunctionHighSampler(JunctionRouteIndex.build(obs,ends),obs,probability=1)
        result=sampler.sample(*query(10))
        self.assertFalse(result['triggered'].any())
        self.assertFalse(result['replaced'].any())

    def test_random_goal_cross_episode_and_returned_label_remain_original(self):
        obs,ends=make_fixture()
        sampler=JunctionHighSampler(JunctionRouteIndex.build(obs,ends),obs,probability=1)
        start,goal,target=query(2)
        goal[0]=int(ends[0]+120)
        target[0]=np.minimum(start[0]+50*np.arange(3,0,-1),goal[0])
        start[1]=40 # E, no junction core query
        target[1]=np.minimum(start[1]+50*np.arange(3,0,-1),goal[1])
        result=sampler.sample(start,goal,target)
        self.assertFalse(result['replaced'].any())
        np.testing.assert_array_equal(result['target_indices'],target)

    def test_zero_probability_and_rng_resume(self):
        obs,ends=make_fixture();index=JunctionRouteIndex.build(obs,ends)
        zero=JunctionHighSampler(index,obs,probability=0)
        before=zero.get_rng_state();r=zero.sample(*query(16))
        self.assertEqual(zero.get_rng_state(),before)
        self.assertFalse(r['replaced'].any());self.assertTrue(r['supported'].all())
        sampler=JunctionHighSampler(index,obs,probability=.5,seed=42)
        sampler.sample(*query(8));state=sampler.get_rng_state()
        expected=sampler.sample(*query(128))
        fresh=JunctionHighSampler(index,obs,probability=.5,seed=0);fresh.set_rng_state(state)
        actual=fresh.sample(*query(128))
        for k in ('start_indices','goal_indices','target_indices','replaced'):
            np.testing.assert_array_equal(actual[k],expected[k])

    def test_index_roundtrip_signature_and_m1_clipping(self):
        obs,ends=make_fixture();index=JunctionRouteIndex.build(obs,ends)
        self.assertEqual(index.signature,JunctionRouteIndex.build(obs,ends).signature)
        altered=obs.copy();altered[0,20]=1
        self.assertNotEqual(index.signature,JunctionRouteIndex.build(altered,ends).signature)
        with tempfile.TemporaryDirectory() as d:
            prefix=pathlib.Path(d)/'index';index.save(prefix);restored=JunctionRouteIndex.load(prefix)
            self.assertEqual(index.signature,restored.signature)
            self.assertEqual(index.geometry,restored.geometry)
            a=JunctionHighSampler(index,obs,probability=1,seed=19).sample(*query(16,M=1))
            b=JunctionHighSampler(restored,obs,probability=1,seed=19).sample(*query(16,M=1))
            np.testing.assert_array_equal(a['target_indices'],b['target_indices'])
            np.testing.assert_array_equal(a['start_indices'],b['start_indices'])
            self.assertTrue(a['replaced'].all())

    def test_m1_and_m3_use_identical_matched_query_draws(self):
        obs,ends=make_fixture();index=JunctionRouteIndex.build(obs,ends)
        a=JunctionHighSampler(index,obs,probability=.5,seed=77).sample(*query(100,M=1))
        b=JunctionHighSampler(index,obs,probability=.5,seed=77).sample(*query(100,M=3))
        for name in ('start_indices','goal_indices','replaced'):
            np.testing.assert_array_equal(a[name],b[name])
        np.testing.assert_array_equal(a['target_indices'][:,0],b['target_indices'][:,-1])

    def test_no_junctions_is_valid_noop(self):
        obs=np.zeros((200,29),np.float32);obs[:,:2]=[4,4]
        index=JunctionRouteIndex.build(obs,np.array([199]))
        self.assertEqual(index.metadata['junctions'],0)
        r=JunctionHighSampler(index,obs,probability=1).sample(*query(4))
        self.assertFalse(r['triggered'].any())
        np.testing.assert_array_equal(r['start_indices'],query(4)[0])

    def test_missing_episode_boundary_rejected(self):
        obs,ends=make_fixture()
        with self.assertRaises(ValueError):JunctionRouteIndex.build(obs,ends[:-1])


if __name__ == '__main__':
    unittest.main()
