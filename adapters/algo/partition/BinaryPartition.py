# -*- coding: utf-8 -*-
"""Implementation of Binary Partition
"""
# Author: Wenjie Li <li3549@purdue.edu>
# License: MIT

from .Node import P_node
from .Partition import Partition
import numpy as np
import copy
import pdb


class BinaryPartition(Partition):
    """
    Implementation of Binary Partition
    """

    def __init__(self, domain=None, node=P_node):
        """
        Initialization of the Binary Partition

        Parameters
        ----------
        domain: list(list)
            The domain of the objective function to be optimized, should be in the form of list of lists (hypercubes),
            i.e., [[range1], [range2], ... [range_d]], where [range_i] is a list indicating the domain's projection on
            the i-th dimension, e.g., [-1, 1]

        node
            The node used in the partition, with the default choice to be P_node.
        """
        if domain is None:
            raise ValueError("domain is not provided to the Binary Partition")
        super(BinaryPartition, self).__init__(domain=domain, node=node)

    # Rewrite the make_children function in the Partition class
    def make_children(self, parent, newlayer=False):
        """
        The function to make children for the parent node with a standard binary partition, i.e., split every
        parent node in the middle. If there are multiple dimensions, the dimension to split the parent is chosen
        randomly

        Parameters
        ----------
        parent:
            The parent node to be expanded into children nodes

        newlayer: bool
            Boolean variable that indicates whether or not a new layer is created

        Returns
        -------

        """

        parent_domain = parent.get_domain()
        '''
        dim = np.random.randint(0, len(parent_domain))
        selected_dim = parent_domain[dim]
        '''
        ranges = [dim_range[1] - dim_range[0] for dim_range in parent_domain]
        dim = np.argmax(ranges)
        selected_dim = parent_domain[dim]

        domain1 = copy.deepcopy(parent_domain)
        domain2 = copy.deepcopy(parent_domain)

        domain1[dim] = [selected_dim[0], (selected_dim[0] + selected_dim[1]) / 2]
        domain2[dim] = [(selected_dim[0] + selected_dim[1]) / 2, selected_dim[1]]

        node1 = self.node(
            depth=parent.get_depth() + 1,
            index=2 * parent.get_index() - 1,
            parent=parent,
            domain=domain1,

        )
        node2 = self.node(
            depth=parent.get_depth() + 1,
            index=2 * parent.get_index(),
            parent=parent,
            domain=domain2,
        )
        parent.update_children([node1, node2])

        new_deepest = []
        new_deepest.append(node1)
        new_deepest.append(node2)

        if newlayer:
            self.node_list.append(new_deepest)
            self.depth += 1
        else:
            self.node_list[parent.get_depth() + 1] += new_deepest

# Rewrite the make_children function in the Partition class
    def make_children_constraint(self, parent, newlayer=False):
        """
        The function to make children for the parent node with a standard binary partition, i.e., split every
        parent node in the middle. If there are multiple dimensions, the dimension to split the parent is chosen
        randomly

        Parameters
        ----------
        parent:
            The parent node to be expanded into children nodes

        newlayer: bool
            Boolean variable that indicates whether or not a new layer is created

        Returns
        -------

        """

        parent_domain = parent.get_domain()

        # 使用对数函数定义权重
        weights = np.array([1 / np.log(i + 2) for i in range(len(parent_domain)-1)])
        weights = weights / weights.sum()  # 归一化权重

        # 根据权重选择 dim
        dim = np.random.choice(len(parent_domain)-1, p=weights)


        #dim = np.random.randint(0,len(parent_domain)-1)
        while(sum(parent_domain[dim]) ==0):
            dim = np.random.randint(0, len(parent_domain) - 1)
        selected_dim = parent_domain[dim]

        domain1 = copy.deepcopy(parent_domain)
        domain2 = copy.deepcopy(parent_domain)

        domain1[dim] = [selected_dim[0], (selected_dim[0] + selected_dim[1]) / 2]
        domain2[dim] = [(selected_dim[0] + selected_dim[1]) / 2, selected_dim[1]]
        for i in range(dim+1, len(parent_domain)-1):
            domain1[i][0] = 0
            domain1[i][1] = 1
            domain2[i][0] = 0
            domain2[i][1] = 1
            for j in range(0,i):
                domain1[i][1] = domain1[i][1] - (domain1[j][0] + domain1[j][1])/2
                domain2[i][1] = domain2[i][1] - (domain2[j][0] + domain2[j][1])/2
            domain1[i][1] = max(0, domain1[i][1])
            domain2[i][1] = max(0, domain2[i][1])
        total1 = 2
        total2 = 2
        for i in range(len(parent_domain)-1):
            total1 -= (domain1[i][0] + domain1[i][1])
            total2 -= (domain2[i][0] + domain2[i][1])
        domain1[-1] = [max(0, total1/2),max(0, total1/2)]
        domain2[-1] = [max(0, total2/2),max(0, total2/2)]




        node1 = self.node(
            depth=parent.get_depth() + 1,
            index=2 * parent.get_index() - 1,
            parent=parent,
            domain=domain1,
        )
        node2 = self.node(
            depth=parent.get_depth() + 1,
            index=2 * parent.get_index(),
            parent=parent,
            domain=domain2,
        )
        parent.update_children([node1, node2])

        new_deepest = []
        new_deepest.append(node1)
        new_deepest.append(node2)

        if newlayer:
            self.node_list.append(new_deepest)
            self.depth += 1
        else:
            self.node_list[parent.get_depth() + 1] += new_deepest

    def make_children_constraint_1(self, parent, newlayer=False):
        """
        The function to make children for the parent node with a standard binary partition, i.e., split every
        parent node in the middle. If there are multiple dimensions, the dimension to split the parent is chosen
        randomly

        Parameters
        ----------
        parent:
            The parent node to be expanded into children nodes

        newlayer: bool
            Boolean variable that indicates whether or not a new layer is created

        Returns
        -------

        """

        parent_domain = parent.get_domain()
        dim = np.random.randint(0,len(parent_domain)-1)
        while(sum(parent_domain[dim]) ==0):
            dim = np.random.randint(0, len(parent_domain) - 1)
        selected_dim = parent_domain[dim]

        domain1 = copy.deepcopy(parent_domain)
        domain2 = copy.deepcopy(parent_domain)

        domain1[dim] = [selected_dim[0], (selected_dim[0] + selected_dim[1]) / 2]
        domain2[dim] = [0, (selected_dim[0] + selected_dim[1]) / 4]
        for i in range(dim+1, len(parent_domain)-1):
            domain1[i][0] = 0
            domain1[i][1] = 2
            domain2[i][0] = 0
            domain2[i][1] = 2
            for j in range(0,i):
                domain1[i][1] = domain1[i][1] - (domain1[j][0] + domain1[j][1])
                domain2[i][1] = domain2[i][1] - (domain2[j][0] + domain2[j][1])
            domain1[i][1] = max(0, domain1[i][1])
            domain2[i][1] = max(0, domain2[i][1])
        total1 = 2
        total2 = 2
        for i in range(len(parent_domain)-1):
            total1 -= (domain1[i][0] + domain1[i][1])
            total2 -= (domain2[i][0] + domain2[i][1])
        domain1[-1] = [max(0, total1/2),max(0, total1/2)]
        domain2[-1] = [max(0, total2/2),max(0, total2/2)]




        node1 = self.node(
            depth=parent.get_depth() + 1,
            index=2 * parent.get_index() - 1,
            parent=parent,
            domain=domain1,
        )
        node2 = self.node(
            depth=parent.get_depth() + 1,
            index=2 * parent.get_index(),
            parent=parent,
            domain=domain2,
        )
        parent.update_children([node1, node2])

        new_deepest = []
        new_deepest.append(node1)
        new_deepest.append(node2)

        if newlayer:
            self.node_list.append(new_deepest)
            self.depth += 1
        else:
            self.node_list[parent.get_depth() + 1] += new_deepest

    def make_children_with_sum_constraint(self, parent, newlayer=False):
        parent_domain = parent.get_domain()
        dim = np.random.randint(0, len(parent_domain))  # 随机选择维度
        selected_dim = parent_domain[dim]

        # 计算当前维度的划分点
        mid_point = (selected_dim[0] + selected_dim[1]) / 2
        domain1 = copy.deepcopy(parent_domain)
        domain2 = copy.deepcopy(parent_domain)
        domain1[dim] = [selected_dim[0], mid_point]
        domain2[dim] = [mid_point, selected_dim[1]]

        # 剩余总和
        sum_dim1 = sum([(d[0] + d[1]) / 2 for d in domain1])
        sum_dim2 = sum([(d[0] + d[1]) / 2 for d in domain2])

        # 剩余调整值
        adjustment1 = 1 - sum_dim1
        adjustment2 = 1 - sum_dim2

        # 计算非划分维度的权重比例
        total_range = sum([d[1] - d[0] for i, d in enumerate(parent_domain) if i != dim])

        for i, d in enumerate(parent_domain):
            if i != dim:
                range_i = d[1] - d[0]  # 当前维度的原始范围
                ratio = range_i / total_range  # 权重比例

                # 按比例调整
                #domain1[i][0] += ratio * adjustment1
                domain1[i][1] += ratio * adjustment1
                #domain2[i][0] += ratio * adjustment2
                domain2[i][1] += ratio * adjustment2

        node1 = self.node(
            depth=parent.get_depth() + 1,
            index=2 * parent.get_index() - 1,
            parent=parent,
            domain=domain1,
        )
        node2 = self.node(
            depth=parent.get_depth() + 1,
            index=2 * parent.get_index(),
            parent=parent,
            domain=domain2,
        )
        parent.update_children([node1, node2])

        new_deepest = []
        new_deepest.append(node1)
        new_deepest.append(node2)

        if newlayer:
            self.node_list.append(new_deepest)
            self.depth += 1
        else:
            self.node_list[parent.get_depth() + 1] += new_deepest

