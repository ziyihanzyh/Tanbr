# -*- coding: utf-8 -*-
"""Implementation of HCT (Azar et al, 2014)
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT

import math
import numpy as np
import random
from .Algo import Algorithm
from .partition.Node import P_node
from .partition.BinaryPartition import BinaryPartition
from .partition.DimensionBinaryPartition import DimensionBinaryPartition
import sys
import pickle
import weakref
#递归深度限制
sys.setrecursionlimit(10000)

def compute_t_plus(x):
    return np.power(2, np.ceil(np.log(x) / np.log(2)))

class HT_node(P_node):

    def __init__(self, depth, index, parent, domain):
        super(HT_node, self).__init__(depth, index, parent, domain)
        self.visited_times = 0
        self.points = self.check_domain_feasibility(domain,[])

    def check_domain_feasibility(self, domain, points):
        dim = len(domain)
        # 如果域本身不可行，直接返回 None
        lower_bounds_sum = sum(bound[0] for bound in domain)
        upper_bounds_sum = sum(bound[1] for bound in domain)

        if lower_bounds_sum > 1 or upper_bounds_sum < 1:
            return []



        def is_valid_point(point):
            """判断点是否为可行解"""
            return (
                    all(constraint[0] <= val <= constraint[1] for val, constraint in zip(point, domain))
                    and np.isclose(np.sum(point), 1.0)
            )

        def generate_point(point_list):
            """生成点并判断是否为可行解"""
            #point_list = []  # 存储可行解的列表
            #points = np.random.dirichlet(np.ones(dim), size=2000)

            # 裁剪点以确保它们在给定的域范围内

            for _ in range(2000):  # 进行 10000 次循环
                point = np.random.dirichlet(np.ones(dim))
                point = np.clip(
                    point,
                    [constraint[0] for constraint in domain],
                    [constraint[1] for constraint in domain]
                )
                point /= np.sum(point)  # 确保点的和为 1

                if is_valid_point(point) and not any(np.allclose(point, p) for p in point_list):
                    point_list.append(point)
                    if len(point_list)>= 50:
                        break

            return point_list

        # 检查输入点是否可行，并加入到 point_list
        point_list = [p for p in points if is_valid_point(p)]
        if len(point_list) > 0:
            point_list = np.array(point_list)
        else:
            point_list = np.empty((0, dim))  # 维度匹配，空数组
        domain = np.array(domain)
        points = np.random.dirichlet(np.ones(dim), size=2000)
        clipped_points = np.clip(points, [constraint[0] for constraint in domain],
                                 [constraint[1] for constraint in domain])
        clipped_points /= clipped_points.sum(axis=1, keepdims=True)
        is_valid = np.all((clipped_points >= domain[:, 0]) & (clipped_points <= domain[:, 1]), axis=1)
        new_points = np.vstack([point_list,clipped_points[is_valid]])
        new_points = np.unique(new_points, axis=0)

        return new_points

    def update(self):
        self.visited_times += 1

    def get_visited_times(self):
        return self.visited_times

class HierTree(Algorithm):
    """
    Implementation of the HCT algorithm
    """

    def __init__(
        self, nu=1, rho=1, c=0.1, delta=0.01, domain_dim = 1, partition=BinaryPartition,p_num=1
    ):
        super(HierTree, self).__init__()
        domain = [[0, 1] for _ in range(domain_dim)]
        if partition is None:
            raise ValueError("Partition of the parameter space is not given.")
        self.partition = partition(domain=domain, node=HT_node)

        self.iteration = 1
        self.nu = nu
        self.rho = rho
        self.delta = delta
        self.c = c
        self.c1 = np.power(rho / (3 * nu), 1.0 / 8)

        self.p_num = p_num
        self.context = []
        self.depth = []

        self.tau_h = [0]  # Threshold on each layer
        self.expand(self.partition.get_root())
        self.reference = weakref.ref(self)



    def __reduce__(self):
        state = self.__dict__.copy()
        state.pop('reference', None)  # 移除不可序列化属性
        return (self.__class__, (), state)

    def get_feasible_leaf_nodes(self):
        """获取所有可行的叶子节点"""
        self.leaf_nodes = []
        nodes = self.partition.get_node_list()
        for layer in nodes:
            # 筛选可行的叶子节点
            self.leaf_nodes.extend([
                node for node in layer
                if node.get_children() is None and len(node.points)>0
            ])
        return self.leaf_nodes

    def get_contexts(self):
        return self.context, self.depth


    def optTraverse(self):
        t_plus = compute_t_plus(self.iteration)
        delta_tilde = np.minimum(1.0 / 2, self.c1 * self.delta / t_plus)
        self.tau_h = [0.0]
        for i in range(1, self.partition.get_depth() + 1):
            self.tau_h.append(
                np.ceil(
                    self.c ** 2
                    * math.log(1 / delta_tilde)
                    * self.rho ** (-2 * i)
                    / self.nu ** 2
                )
            )

        curr_node = self.partition.get_root()
        path = [curr_node]

        while (
            curr_node.get_visited_times() >= self.tau_h[curr_node.get_depth()]
            and curr_node.get_children() is not None
        ):
            children = curr_node.get_children()
            maxchild = children[0]
            for child in children[1:]:

                if child.get_b_value() >= maxchild.get_b_value():
                    maxchild = child

            curr_node = maxchild
            path.append(maxchild)

        return curr_node, path



    def expand(self, parent):
        if len(parent.points)<1:
            return

        if parent.get_depth() >= self.partition.get_depth():
            self.partition.make_children(parent=parent, newlayer=True)
        else:
            self.partition.make_children(parent=parent, newlayer=False)

        # 检查新生成的子节点的可行性
        if parent.get_children():
            for child in parent.get_children():
                child.points = child.check_domain_feasibility(child.get_domain(), parent.points)

        nodes = self.get_feasible_leaf_nodes()
        total_points = len(nodes) * self.p_num

        # 预分配内存
        context = np.empty((total_points, nodes[0].points[0].shape[0]), dtype=np.float32)  # 假设每个点是一个向量
        depth = np.empty(total_points, dtype=int)

        # 填充数据
        index = 0
        for n in nodes:
            selected_points = random.choices(n.points, k=self.p_num)  # 选择点
            context[index:index + self.p_num] = selected_points  # 批量赋值
            depth[index:index + self.p_num] = [n.depth] * self.p_num  # 批量赋值
            index += self.p_num
        self.context = context
        self.depth = depth

    def update(self, node):
        node.update()
        self.iteration += 1

        node_depth = node.get_depth()


        t_plus = compute_t_plus(self.iteration)
        delta_tilde = np.minimum(1.0 / 2, self.c1 * self.delta / t_plus)

        tau_h = np.ceil(
                    self.c ** 2
                    * math.log(1 / delta_tilde)
                    * self.rho ** (-2 * node_depth)
                    / self.nu ** 2)
        if node.get_visited_times() >= tau_h:
            self.expand(node)


    def pull(self, time):
        self.curr_node, self.path = self.optTraverse()
        return self.curr_node.get_cpoint()

    def receive_reward(self, node_num,reward):
        self.update(self.leaf_nodes[node_num])

    def get_last_point(self):
        return self.pull(0)


