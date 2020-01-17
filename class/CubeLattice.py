import sys
import itertools
import numpy as np
import operator as op
from functools import reduce
import reference as rf
import PolyLattice as pl
import collections as cln
import time

class CubeLattice:

    def __init__(self, lb, ub):
        if len(ub)!=len(lb):
            print('The lengths of ub and lb are not equal')
            sys.exit(1)

        self.dim = len(lb)
        self.lb = lb
        self.ub = ub
        self.bs = np.array([lb,ub]).transpose()
        self.vertex = self.compute_vertex(lb, ub)
        self.lattice, self.id_vals, self.vertex_ref, self.ref_vertex = self.initial_lattice()
        for m in range(1,self.dim):
            self.single_dim_face(m)

        self.M = np.eye(self.dim)
        self.b = np.zeros((self.dim,1))

    def to_poly_lattice(self):
        return pl.PolyLattice(self.lattice, self.ref_vertex, self.vertex, self.dim, self.M, self.b)

    def initial_lattice(self):
        lattice = []
        id_vals = []
        vertex_ref = cln.OrderedDict()
        ref_vertex = cln.OrderedDict()
        n = self.dim
        for m in range(self.dim):
            num = 2**(n-m)*self.ncr(n,m)
            d = cln.OrderedDict()
            val = cln.OrderedDict()
            for i in range(num):
                id = rf.Refer(i)
                d.update({id:[set(),set()]})
                val.update({id: [[],[]]})
                if m == 0:
                    vertex_ref.update({tuple(self.vertex[i]):id})
                    ref_vertex.update({id: self.vertex[i]})
            lattice.append(d)
            id_vals.append([])
        return lattice, id_vals, vertex_ref, ref_vertex

    def compute_vertex(self, lb, ub):
        # compute vertex
        V = []
        for i in range(len(ub)):
            V.append([lb[i], ub[i]])

        return np.array(list(itertools.product(*V)))

    # update lattice of m_face
    def single_dim_face(self, m):
        num = 2 ** (self.dim - m) * self.ncr(self.dim, m)
        Varray = self.vertex
        ref_m = list(self.lattice[m].keys())
        ref_m_1 = list(self.lattice[m-1].keys())

        id_vals_temp = cln.OrderedDict()

        nlist = list(range(len(self.lb)))
        element_id_sets = list(itertools.combinations(nlist, self.dim-m))
        c = 0
        for element_id in element_id_sets:
            # start_time = time.time()
            elem_id_m = np.array(element_id)
            vals = [list(self.bs[e,:]) for e in elem_id_m]
            faces = np.array(list(itertools.product(*vals)))

            diff_elem = np.setdiff1d(np.array(range(self.dim)), elem_id_m)

            for f in faces:
                f_m = np.ones((self.dim))*100
                for i in range(len(elem_id_m)):
                    f_m[elem_id_m[i]] = f[i]
                k_m = tuple(np.concatenate((elem_id_m, f_m)))
                id_m = ref_m[c]
                id_vals_temp.update({k_m: id_m})

                for i in diff_elem:
                    elem_id_m_1 = np.copy(elem_id_m)
                    elem_id_m_1 = np.sort(np.append(elem_id_m_1, i))
                    f_m_1 = np.copy(f_m)
                    # upper bound
                    f_m_1[i] = self.ub[i]
                    k_m_1 = tuple(np.concatenate((elem_id_m_1, f_m_1)))
                    if m!=1:
                        id_m_1 = self.id_vals[m - 1][k_m_1]
                    else:
                        id_m_1 = self.vertex_ref[tuple(f_m_1)]

                    self.lattice[m][ref_m[c]][0].add(id_m_1)
                    self.lattice[m - 1][id_m_1][1].add(ref_m[c])

                    # lower bound
                    f_m_1[i] = self.lb[i]
                    k_m_1 = tuple(np.concatenate((elem_id_m_1, f_m_1)))
                    if m != 1:
                        id_m_1 = self.id_vals[m - 1][k_m_1]
                    else:
                        id_m_1 = self.vertex_ref[tuple(f_m_1)]

                    self.lattice[m][ref_m[c]][0].add(id_m_1)
                    self.lattice[m - 1][id_m_1][1].add(ref_m[c])

                c = c+1

        self.id_vals[m] = id_vals_temp

        if c!=num:
            print('Computation is wrong')
            sys.exit(1)


    def ncr(self, n, r):
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return int(numer / denom)


    # def face_vertex(self, m, aset):
    #     ref_m = list(self.lattice[m].keys())
    #     ref_m_1 = list(self.lattice[m-1].keys())
    #     V = []
    #     if m == 1:
    #         vs = []
    #         for s in aset:
    #             vs.append(self.ref_vertex[s])
    #         return vs
    #     else:
    #         new_set =set()
    #         for s in aset:
    #             new_set.update(self.lattice[m-1][s][0])
    # 
    #     V.extend(self.face_vertex(m-1, new_set))
    #     return V

    # def test_result(self):
    #     hss = []
    #     for m in range(1,len(self.lattice)):
    #         mface = set()
    #         for k, v in self.lattice[m].items():
    #             vs = self.face_vertex(m, v[0])
    #             for v in vs:
    #                 id = self.vertex.index(v)
    #                 mface.add(id)
    #             hss.append(mface)
    #             mface = set()
    #     return hss

# # test
# lb = [-1,-1,-1]
# ub = [1,1,1]
# Vertex = CubeLattice(lb,ub)
# m=1
# ref_m = list(Vertex.lattice[m].keys())
# print(Vertex.face_vertex(m, Vertex.lattice[m][ref_m[7]][0]))
# print(Vertex.test_result())
# print(Vertex.vertex_ref)
# from scipy.spatial import ConvexHull
# points = np.random.rand(200, 5)
# hull = ConvexHull(points)
# print(hull.vertices)