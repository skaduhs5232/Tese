import numpy as np

class Percolation:
    def __init__(self, arr):
        self.arr = arr

        self.nrows, self.ncols = self.arr.shape
        self.n = self.nrows * self.ncols

        print(self.nrows, self.ncols, self.n)

        self.parent = np.arange(self.n, dtype = np.int64)
        self.rank = np.zeros(self.n, dtype = np.uint8)
    
    def union(self, x, y):
        self.link(self.find_set(x), self.find_set(y))
    
    def link(self, x, y):
        if self.rank[x] > self.rank[y]:
            self.parent[y] = x
        else:
            self.parent[x] = y
            if self.rank[x] == self.rank[y]:
                self.rank[y] += 1
    
    def find_set(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find_set(self.parent[x])
        return self.parent[x]
    
    def site_percolation(self):
        self.half = self.ncols // 2

        for k in range(self.n):
            j = k // self.ncols
            i = k % self.ncols
            if self.arr[j, i]:
                ir = (i + 1) % self.ncols
                if self.arr[j, ir]:
                    kr = j * self.ncols + ir
                    self.union(k, kr)
                
                jd = j + 1
                if jd < self.nrows and self.arr[jd, i]:
                    kd = k + self.ncols
                    self.union(k, kd)
        
        last = (self.nrows - 1) * self.ncols
        for i in range(self.half):
            if self.arr[0, i] and self.arr[0, i + self.half]:
                self.union(i, i + self.half)
            
            if self.arr[self.nrows - 1, i] and self.arr[self.nrows - 1, i + self.half]:
                self.union(last + i, last + (i + self.half))
        
        for k in range(self.n):
            self.parent[k] = self.find_set(k)