import heapq
import numpy as np
from radbm.search.base import BaseSDS

class GridSearch(BaseSDS):
    """
    An exact symmetric search (query and documents are in the same space)
    GridSearch separates the space into a grid and fill each cell with
    documents using a dictionary (cell -> documents). For retrieval, the
    the cells are searched in order of distance and document are yielded
    when it is the closest found so far and when the next nearest cell is
    farther than the document (this is certifying there is no document
    closer).
    
    Parameters
    ----------
    bounds : np.ndarray
        shape=(2,d) with d the dimension of the data (query/documents)
        min_bounds, max_bounds = bounds are the bounds of the grid.
    divisions : np.ndarray, dtype=int
        shape=(d,) with d the dimension of the data (query/documents)
        the number of cell along each axis. 
    """
    def __init__(self, bounds, divisions):
        self.bounds = bounds
        self.min_bounds, self.max_bounds = bounds
        self.delta_bounds = self.max_bounds - self.min_bounds
        self.divisions = divisions
        self.grid = dict()
        
        #used in multiple places, a range of a dimensions
        self.range = np.arange(len(divisions))
        
        #width2 is used to compute L2 given the scalled data
        self.width2 = (self.delta_bounds/self.divisions)**2
        
        #directions are used for the graph search version of the cell generator
        eye = np.eye(len(divisions))
        self.directions = np.concatenate([eye, -eye])

    def compute_scaled_points(self, points):
        """
        Rescale point(s)
        
        Parameters
        ----------
        points : np.ndarray
            A point or a batch of points (query/document)
        
        Returns
        -------
        rescalled_points : np.ndarray
            rescalled_point[i] in [0, self.divisions[i]]
        """
        return self.divisions*(points - self.min_bounds)/self.delta_bounds

    def compute_cells(self, points):
        """
        Returns
        -------
        cell : np.ndarray, dtype=int
            The cells of the corresponding points referended by the
            closest to the origin corner.
        """
        scaled_points = self.compute_scaled_points(points)
        return np.floor(scaled_points).astype(np.int)

    def compute_cell_relative_positions(self, points):
        """
        Gives the position relative to the cell of the points
        Returns
        -------
        cell_pos : np.ndarray
            cell_pos in [0, 1]
        """
        scaled_points = self.compute_scaled_points(points)
        cells = np.floor(scaled_points).astype(np.int)
        return scaled_points - cells

    def is_cell_inbound(self, cell):
        return all(0 <= c < d for c,d in zip(cell, self.divisions))
    
    def is_points_inbound(self, points):
        inside = np.logical_and(self.min_bounds <= points, points < self.max_bounds)
        return np.all(inside)
    
    def batch_insert(self, points, indexes):
        if not self.is_points_inbound(points):
            raise ValueError('points out of bounds')
            
        #grouping points by cell
        grid = dict()
        cells = self.compute_cells(points)
        for cell, point, index in zip(cells, points, indexes):
            tcell = tuple(cell)
            if tcell in grid:
                grid[tcell][0].append(point)
                grid[tcell][1].append(index)
            else: grid[tcell] = [[point], [index]]
                
        #updating self.grid
        for tcell, (new_points, new_indexes) in grid.items():
            if tcell in self.grid:
                points, indexes = self.grid[tcell]
                self.grid[tcell] = [
                    np.concatenate([points, new_points]),
                    np.concatenate([indexes, new_indexes]),
                ]
            else: 
                self.grid[tcell] = [
                    np.array(new_points),
                    np.array(new_indexes)
                ]
        
        return self

    def point_relative_cells_dists(self, delta_x, relative_cells):
        signature = (0<relative_cells).astype(int)
        a = delta_x[signature, self.range]
        mask = (0!=relative_cells)
        tmp = (a + (np.abs(relative_cells)-1))**2 * self.width2
        return np.sqrt((tmp*mask).sum(axis=1))

    def graphsearch_cell_generator(self, point):
        scaled_point = self.compute_scaled_points(point)
        #zero_cell = compute_cells(point)
        zero_cell = np.floor(scaled_point).astype(np.int)
        #zero_cell_pos = compute_cell_relative_positions(point)
        zero_cell_pos = scaled_point - zero_cell
        delta_x = np.stack([zero_cell_pos, 1 - zero_cell_pos])
        tzero_cell =  tuple(zero_cell)
        zero = tuple(0 for _ in self.range)
        heap = [(0, zero)]
        expanded_set = {zero}
        while heap:
            dist, tcell = heapq.heappop(heap)
            yield dist, tuple((zero_cell+tcell).astype(int))
            ncells = self.directions + tcell
            dists = self.point_relative_cells_dists(delta_x, ncells)
            for ncell, dist in zip(ncells, dists):
                tncell = tuple(ncell)
                if tncell not in expanded_set and self.is_cell_inbound(tncell+zero_cell):
                    expanded_set.add(tncell)
                    heapq.heappush(heap, (dist, tncell))

    def compute_cells_dist(self, point):
        pos = self.compute_scaled_points(point)
        cell = pos.astype(int)
        cell_pos = pos - cell
        ndim = len(self.divisions)
        dists = np.array(0).reshape(*(1 for dim in range(ndim)))
        for dim, div, w2, k, d in zip(range(ndim), self.divisions, self.width2, cell, cell_pos):
            right = [i+(1-d) for i in range(div-k-1)]
            left = [(k-i-1)+d for i in range(k)]
            view = tuple(1 if i!=dim else -1 for i in range(ndim))
            axis_dists = np.concatenate([left, [0], right])**2
            dists = dists + axis_dists.reshape(view)*w2
        return np.sqrt(dists)

    def bruteforce_cell_generator(self, point):
        dists = self.compute_cells_dist(point)
        flatten_dists = dists.flatten()
        #mlog(m) with m the number of cell
        argsort = np.argsort(flatten_dists)
        sorted_dists = flatten_dists[argsort]
        sorted_cells = np.stack(np.unravel_index(argsort, self.divisions), axis=1)
        for dist, cell in zip(sorted_dists, sorted_cells):
            yield dist, tuple(cell)

    def itersearch(self, point, cell_generator='bruteforce',
                   max_dist=np.inf, max_items=np.inf,
                   yield_point=False, yield_dist=False):
        """
        Generator
        
        Parameters
        ----------
        point : np.ndarray
        cell_generator : str
            should be bruteforce or graphsearch
        max_dist : number
        max_items : number
        yield_point : bool
        yield_dist : bool
        
        Yields
        ------
        index : int
            if yield_point is False and yield_dist is False
        point : np.ndarray
            if yield_point is True and yield_dist is False
        (dist, index) : (float, int)
            if yield_point is False and yield_dist is True
        (dist, point) : (float, np.ndarray)
            if yield_point is True and yield_dist is True
        """
        if point.ndim != 1:
            raise ValueError('expected flatt array, got ndim={}'.format(point.ndim))
        
        if not self.is_points_inbound(point):
            raise ValueError('point out of bounds')
            
        if cell_generator == 'bruteforce':
            cell_generator = self.bruteforce_cell_generator(point)
        elif cell_generator == 'graphsearch':
            cell_generator = self.graphsearch_cell_generator(point)
        else:
            msg = 'cell_generator must be bruteforce or graphsearch, got {}'
            raise ValueError(msg.format(cell_generator))
                    
        if max_items <= 0 or max_dist < 0:
            #nothing to do
            return
        
        heap = []
        item_count = 0
        for dist, cell in cell_generator:
            while heap and heap[0][0] < dist and heap[0][0] < max_dist:
                d, p, i = heapq.heappop(heap)
                x = p if yield_point else i
                yield (d, x) if yield_dist else x
                item_count += 1
                if max_items <= item_count:
                    #yielded enough items
                    return   
            if max_dist < dist:
                #all further cells are farther
                #we already yielded every documents
                #closer than max_dist
                return
            if cell in self.grid:
                points, indexes = self.grid[cell]
                dists = np.sqrt(((point-points)**2).sum(axis=1))
                for d, p, i in zip(dists, points, indexes):
                    #the next lines are for debugging
                    #if d < dist:
                    #    msg = (
                    #        'something went wrong, found a point {} at '
                    #        'distance {:.4f} in the cell {} at distance {:.4f}'
                    #    )
                    #    raise ValueError(msg.format(p, d, cell, dist))
                    heapq.heappush(heap, (d, p, i))
        #empty the heap
        while heap and heap[0][0] < max_dist:
            d, p, i = heapq.heappop(heap)
            x = p if yield_point else i
            yield (d, x) if yield_dist else x
            item_count += 1
            if max_items <= item_count:
                #yielded enough items
                return