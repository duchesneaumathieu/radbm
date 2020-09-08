import unittest
import numpy as np
from radbm.search.gridsearch import GridSearch

class TestGridSearch(unittest.TestCase):
    def setUp(self):
        self.bounds = np.array([[-1,-2,-3], [3,2,1]])
        self.divisions = np.array([4, 2, 8])
        self.gs = GridSearch(self.bounds, self.divisions)
    
    def test_point_arithmetic(self):
        points = np.array([
            [0.6, 1.5, -2.1],
            [0, 1, 0],
        ])
        scaled_points = self.gs.compute_scaled_points(points)
        cells = self.gs.compute_cells(points)
        pos = self.gs.compute_cell_relative_positions(points)
        self.assertTrue((pos==(scaled_points-cells)).all())
        
        self.assertTrue(self.gs.is_cell_inbound((0,0,0)))
        self.assertFalse(self.gs.is_cell_inbound((0,2,0)))
        
        self.assertTrue(self.gs.is_points_inbound(np.array([0,0,0])))
        self.assertTrue(self.gs.is_points_inbound(np.array([-1,-2,-3])))
        self.assertFalse(self.gs.is_points_inbound(np.array([-4,6,2])))
        self.assertFalse(self.gs.is_points_inbound(np.array([3,0,0])))
        
        #graphsearch_cell_generator specific
        min_bounds, max_bounds = self.bounds
        width = (max_bounds-min_bounds)/self.divisions
        self.assertEqual(list(width**2), list(self.gs.width2))
        delta_x = np.stack([pos[0], 1-pos[0]])
        rel_cells = np.array([
            [0,0,0],
            [1,0,0],
            [3,0,0],
            [0,0,2],
            [0,0,-2],
            [3,0,-2],
        ])
        expected_dists2 = [
            0,
            (delta_x[1,0]*width[0])**2,
            ((delta_x[1,0] + 2)*width[0])**2,
            ((delta_x[1,2] + 1)*width[2])**2,
            ((delta_x[0,2] + 1)*width[2])**2,
            ((delta_x[1,0] + 2)*width[0])**2 + ((delta_x[0,2] + 1)*width[2])**2,
        ]
        dists2 = self.gs.point_relative_cells_dists(delta_x, rel_cells)**2
        self.assertEqual(expected_dists2, list(dists2))
        
        #bruteforce_cell_generator specific
        #implicitly tested
        #self.gs.compute_cells_dist(self.points[0])
        
    def test_cell_generators(self):
        point = np.array([0.12, -1.1, .74])
        g1 = list(self.gs.graphsearch_cell_generator(point))
        g2 = list(self.gs.bruteforce_cell_generator(point))
        self.assertEqual(g1, g2)
        
    def test_retrieval(self):
        #out of bound point
        out_point = np.array([-4,0,0])
        with self.assertRaises(ValueError):
            self.gs.insert(out_point, 1)
            
        with self.assertRaises(ValueError):
            next(self.gs.itersearch(out_point))
            
        rng = np.random.RandomState(0xcafe)
        min_bounds, max_bounds = self.bounds
        delta = max_bounds - min_bounds
        
        #test double insert
        points1 = rng.uniform(0,1,(1000,3))*delta + min_bounds
        self.gs.batch_insert(points1, range(1000))
        points2 = rng.uniform(0,1,(1000,3))*delta + min_bounds
        self.gs.batch_insert(points2, range(1000,2000))
        points = np.concatenate([points1, points2])
        
        point = np.array([0.12, -1.1, .74])
        dists2 = ((point-points)**2).sum(axis=1)
        argsort = list(np.argsort(dists2))
        
        #ndim!=1 error
        with self.assertRaises(ValueError):
            next(self.gs.itersearch(point[None]))
        
        #wrong cell_generator error
        with self.assertRaises(ValueError):
            next(self.gs.itersearch(point, cell_generator='something'))
        
        #test exactness
        graphsearch = list(self.gs.itersearch(
            point, cell_generator='graphsearch', 
            max_dist=np.inf, max_items=np.inf
        ))
        self.assertEqual(graphsearch, argsort)
        
        bruteforce = list(self.gs.itersearch(
            point, cell_generator='bruteforce', 
            max_dist=np.inf, max_items=np.inf
        ))
        self.assertEqual(bruteforce, argsort)
        
        #test max_items==0
        self.assertEqual(list(self.gs.itersearch(point, max_items=0)), list())
        
        #test max_dist==0
        self.assertEqual(list(self.gs.itersearch(point, max_dist=0)), list())
        
        #test max_items>0
        self.assertEqual(len(list(self.gs.itersearch(point, max_items=42))), 42)
        
        #test max_dist>0
        items = list(self.gs.itersearch(point, max_dist=1))
        self.assertTrue(np.sqrt(dists2[items[-1]]) < 1)
        self.assertTrue(np.sqrt(dists2[argsort[np.where(argsort==items[-1])[0][0]+1]] >= 1))
        
        #test max_items>0 and all cell generated
        self.assertEqual(len(list(self.gs.itersearch(point, max_items=1999))), 1999)