from typing import List, Tuple, Any, Callable, Optional
from math import ceil, log2
import numpy as np



class SumTree:
    """
    Given a list of (data, td_error) tuples, this structure creates and stores a sum_tree of DataNodes.
    It is assumed that each tuple contains the td_error value.
    """
    def __init__(self, capacity:int, alpha: float):
        
        self._epsilon = 1e-5  # a small value to ensure no division by 0
        self._alpha = alpha
        self._capacity = capacity
        
        self._len_sum_tree = int(2 * 2**ceil(log2(self._capacity)) - 1)
        
        self._arr:List[Tuple] = [ (None, 0) for _ in range(self._capacity)]
        
        # sumTree arrs
        self._cumul_priority = np.array([0.0 for _ in range(self._len_sum_tree)], dtype=np.float32)
        self._cumul_inv_priority = np.array([0.0 for _ in range(self._len_sum_tree)], dtype=np.float32)
        self._min_priority = np.array([0.0 for _ in range(self._len_sum_tree)], dtype=np.float32)
        

        # Build tree and find min priority node
        self.build_sum_tree(0, self._capacity - 1, 0)
        self._arr_min_priority_index = self.find_min_priority_index()
        
        self._arr_to_tree_ind = np.array([ self._arr_to_tree_index_helper(arr_ind) for arr_ind in range(self._capacity)],  dtype=np.int32)

    def build_sum_tree(self, start: int, end: int, tree_ind: int):
        """
        Recursively initializes the entire sum‐tree array (`sum_arr`) over the range [start, end].

        For each leaf node (when start == end), this sets:
            - cumul_priority = ε
            - cumul_inv_priority = 1/ε
            - min_priority = ε

        For each internal node, this sets:
            - cumul_priority = left_child.cumul_priority + right_child.cumul_priority
            - cumul_inv_priority = left_child.cumul_inv_priority + right_child.cumul_inv_priority
            - min_priority = min(left_child.min_priority, right_child.min_priority)

        Args:
            start   (int): Left index of the segment (inclusive).
            end     (int): Right index of the segment (inclusive).
            tree_ind   (int): Position in `sum_arr` where this node’s values should be written.

        Returns:
            Tuple: cumulative_priority, cumulative_inverse_priority, and minimum_priority after initialization.
        """
        if start == end:  # Leaf node
            self._cumul_priority[tree_ind] = self._epsilon
            self._cumul_inv_priority[tree_ind] = 1.0/self._epsilon
            self._min_priority[tree_ind] = self._epsilon
            return self._epsilon, 1.0/self._epsilon, self._epsilon

        mid = (start + end) // 2
        left_child = self.build_sum_tree(start, mid, 2 * tree_ind + 1)
        right_child = self.build_sum_tree(mid + 1, end, 2 * tree_ind + 2)

        cumul_p = left_child[0] + right_child[0]
        cumul_inv_p = left_child[1] + right_child[1]
        min_p = min(left_child[2], right_child[2])
        
        self._cumul_priority[tree_ind] = cumul_p
        self._cumul_inv_priority[tree_ind] = cumul_inv_p
        self._min_priority[tree_ind] = min_p

        return cumul_p, cumul_inv_p, min_p

    def _arr_to_tree_index_helper(self, arr_ind: int):
        """
        Maps an index from the original data array (`arr`) to its corresponding index in the sum tree.

        This function traverses the implicit binary tree structure of the sum tree to find the index
        that represents the leaf node associated with `arr[arr_ind]`.

        Args:
            arr_ind (int): Index in the original data array (`arr`).

        Returns:
            int: Corresponding index in sum tree representing the leaf node for `arr[arr_ind]`.
        """
        tree_ind = 0
        start = 0
        end = self._capacity - 1

        while start != end:
            mid = (start + end) // 2
            if arr_ind > mid:
                start = mid + 1
                tree_ind = 2 * tree_ind + 2  # Move to the right child
            else:
                end = mid
                tree_ind = 2 * tree_ind + 1  # Move to the left child
        return tree_ind
                
    def find_min_priority_index(self):
        """
        Finds the leaf node in sum tree whose `min_priority` is minimal, and returns
        both that leaf's index in sumtree and the corresponding index in the original array.

        Returns: The corresponding index in the original data array `self._arr` (same as the leaf's segment index).

        Raises:
            ValueError: If neither the left nor right child contains the parent's `min_priority`,
                        indicating an inconsistent tree state.
        """
        start = 0
        end = self._capacity - 1
        tree_ind = 0

        while True:
            # If we’re at a leaf (start == end) and its min_priority equals its cumul_priority, stop.
            if (self._min_priority[tree_ind] == self._cumul_priority[tree_ind] ) and (start == end):
                return start

            left_child = 2 * tree_ind + 1
            right_child = 2 * tree_ind + 2
            mid = (start + end) // 2

            # If the left child’s min_priority matches, go left
            if self._min_priority[tree_ind] == self._min_priority[left_child]:
                tree_ind = left_child
                end = mid

            # Otherwise, if the right child’s min_priority matches, go right
            elif self._min_priority[tree_ind] == self._min_priority[right_child]:
                tree_ind = right_child
                start = mid + 1

            # If neither child matches, the tree is inconsistent
            else:
                raise ValueError(
                    f"Invalid sum tree structure at index {tree_ind}: "
                    f"min_priority ({self._min_priority[tree_ind]}) not found in "
                    f"left ({self._min_priority[left_child]}) or right "
                    f"({self._min_priority[right_child]}) child."
            )
        
    def point_update(self, arr_ind: int, input_tuple: Tuple[Any, float]):
        """
        Update the data and TD-error at a given index, recompute its priority, and adjust the sum tree accordingly.

        Args:
            arr_ind (int): Index in the original array to update.
            input_tuple (Tuple[Any, float]): New (data, td_error) pair for that index.

        Returns:
            None
        """
        # Update the original array
        data, td_error = input_tuple
        self._arr[arr_ind] = (data, td_error)

        # compute priority and update relevant nodes
        priority = (abs(td_error) + self._epsilon)**self._alpha
        
        tree_ind = self._arr_to_tree_ind[arr_ind] # The corresponding node index in the sum tree

        # difference between new and old priorities
        delta_priority = priority - self._cumul_priority[tree_ind]
        delta_inv_priority = 1.0/priority - self._cumul_inv_priority[tree_ind]

        # Update the leaf node with the new data
        self._cumul_priority[tree_ind] = priority
        self._cumul_inv_priority[tree_ind] = 1.0/priority
        self._min_priority[tree_ind] = priority

        # Propagate the changes up the sum tree
        propagated_ind = tree_ind
        while propagated_ind > 0:
            parent = (propagated_ind - 1) // 2  # Move to the parent node
            self._cumul_priority[parent] += delta_priority
            self._cumul_inv_priority[parent] += delta_inv_priority

            # Restore min-priority tracking
            sibling_index = propagated_ind - 1 if propagated_ind % 2 == 0 else propagated_ind + 1

            if sibling_index < self._len_sum_tree:
                self._min_priority[parent] = min(self._min_priority[propagated_ind] , self._min_priority[sibling_index])
            else: raise ValueError(f"Sibling node at index {sibling_index} does not exist for node {propagated_ind}. Tree structure may be inconsistent.")

            propagated_ind = parent

        # If the updated node had the minimum priority, find the new min-priority index
        if arr_ind == self._arr_min_priority_index: self._arr_min_priority_index = self.find_min_priority_index()

    def range_update(self, arr_inds: List[int], input_tuple_list: List[Tuple[Any, float]]):
        """
        Batch-update multiple leaves and their ancestors in the sum tree.

        Overwrite each leaf's (data, td_error) → new priority = (|td_error| + ε)^α,
        then percolate up any changed leaves until reaching the root.

        Args:
            arr_inds (List[int]): Indices in the data array to update.
            input_tuple_list (List[Tuple[Any, float]]): New (data, td_error) pairs,
                one per index in `arr_inds`.

        Returns:
            None
        """
        num_updates = len(arr_inds)
        _, td_error = zip(*input_tuple_list)
        td_error = np.array(td_error, dtype=np.float32)

        # 1) Update leaves
        priorities = (np.abs(td_error) + self._epsilon) ** self._alpha

        # together these 2 structures pretend to be an orderset set
        fifo_queue = [] # preserves the order
        unique_set = set() # preserves the uniqueness
        
        for i in range(num_updates):
            leaf = self._arr_to_tree_ind[arr_inds[i]]
            if leaf in unique_set: continue # no need to add a copy
            
            # update the data array
            self._arr[arr_inds[i]] = input_tuple_list[i]
            new_p = priorities[i]

            self._cumul_priority[leaf]     = new_p
            self._cumul_inv_priority[leaf] = 1.0 / new_p
            self._min_priority[leaf]       = new_p

            fifo_queue.append(leaf)
            unique_set.add(leaf)

        # Changes to the relevant leaves have been made in the for-loop.
        # 2) Propagate the changes up the sum tree to the root O(k log(N))
        # for each index, its parent is updated
            
        while True:
            # 2.a) get the new index to process
            curr_tree_ind = fifo_queue.pop(0)
            
            # 2.b) If already processed, get another index
            if not curr_tree_ind in unique_set: continue
            
            # 2.c) Otherwise remove from the set, since processing right now
            unique_set.remove(curr_tree_ind)
            
            # 2.d) # root has no parent. all nodes processed, since a child of root must have updated root
            if curr_tree_ind ==0: break 
            
            # 2.e) otherwise calculate parent and sibling indexes and process the parent
            parent_ind = (curr_tree_ind -1)//2
            sibling_ind = curr_tree_ind - 1 if curr_tree_ind % 2 == 0 else curr_tree_ind + 1
            
            if sibling_ind >= self._len_sum_tree:
                raise ValueError(f"Sibling node at index {sibling_ind} does not exist for node {parent_ind}. Tree structure may be inconsistent.")
            
            # make the updates
            self._cumul_priority[parent_ind]        = self._cumul_priority[curr_tree_ind] + self._cumul_priority[sibling_ind]
            self._cumul_inv_priority[parent_ind]    = self._cumul_inv_priority[curr_tree_ind] + self._cumul_inv_priority[sibling_ind]
            self._min_priority[parent_ind]          = min( self._min_priority[curr_tree_ind], self._min_priority[sibling_ind]  ) 
            
            # 2.f) remove the sibling from the set, if present. Does not matter if present in the queue, since the set decides if the index would be processed
            if sibling_ind in unique_set:
                unique_set.remove(sibling_ind)
            
            # 2.g) add parent in both set and queue. If sibling was processed before then this node would have been processed, since 2.f removes the sibling from the set  
            fifo_queue.append(parent_ind) # repeat up the tree
            unique_set.add(parent_ind)
            
        # 3) Assume that one of the updates was for min priority and just re-evaluate the indexes O(log(n))
        self._arr_min_priority_index = self.find_min_priority_index()

    def sample_data_by_priority(self, cumul_targets: np.ndarray, num_samples):
        """
        Sample multiple leaves from the Sum‐Tree given an array of cumulative priorities.

        For each entry in `cumul_targets`, this method walks down the binary
        Sum‐Tree to find the corresponding leaf whose priority-weighted range
        contains that target. All `num_samples` traversals proceed in parallel.

        Args:
            cumul_targets (np.ndarray of shape (num_samples,)):
                Each value in [0, total_priority_sum) used to select a leaf.
            num_samples (int):
                Number of leaves to sample; must equal len(cumul_targets).

        Returns:
            Tuple of three elements:
            - arr_indices (np.ndarray of shape (num_samples,), dtype=int32):
                Index in `self._arr` for each sampled leaf.
            - out_priorities (np.ndarray of shape (num_samples,), dtype=float32):
                The priority value stored at each sampled leaf node.
            - data_list (list of length num_samples):
                The payloads: for each index i, `data_list[i] == self._arr[arr_indices[i]][0]`.

        Raises:
            IndexError:
                If a leaf’s start/end indices disagree (i.e. start != end).
            ValueError:
                If updated cumulative targets become slightly negative (< –1e-10).
        """
        
        # Initialize global state arrays
        tree_inds = np.zeros(num_samples, dtype=np.int32)
        starts = np.zeros(num_samples, dtype=np.int32)
        ends = np.full(num_samples, self._capacity - 1, dtype=np.int32)
        active = np.ones(num_samples, dtype=bool)
        out_arr_idxs = np.zeros(num_samples, dtype=np.int32)
        out_priorities = np.zeros(num_samples, dtype=np.float32)
        

        while np.any(active):
            # Get global indices of active elements
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:break
            
            # Compute child indices for active elements
            left_childs_ind = 2 * tree_inds[active_indices] + 1
            right_childs_ind = 2 * tree_inds[active_indices] + 2
            
            # Identify leaf nodes: out-of-bound OR in-bound with zero priority
            out_of_bound = (left_childs_ind >= self._len_sum_tree)
            leaf_condition = out_of_bound.copy()  # Start with out-of-bound condition
        
            # Check in-bound nodes for zero priority
            if np.any(~out_of_bound):
                # leaf condition for in-bound nodes = inv_prio
                leaf_condition[~out_of_bound] = (self._cumul_inv_priority[left_childs_ind[~out_of_bound]] == 0.0)  # Mark in-bound zero-prio as leaves
            
            # Process leaf nodes
            if np.any(leaf_condition):
                
                # Validate leaf segments
                bad_mask = starts[active_indices[leaf_condition]] != ends[active_indices[leaf_condition]]
                if np.any(bad_mask):
                    bad_indices = active_indices[leaf_condition][bad_mask]
                    error_msg = "\n".join(
                        f"Index {i}: start={starts[i]}, end={ends[i]}" 
                        for i in bad_indices
                    )
                    raise IndexError(f"Segment mismatch (array len={self._capacity}):\n{error_msg}")
                
                # Store results and deactivate
                
                # This indexing is for self.arr_inds
                out_arr_idxs[active_indices[leaf_condition]] = starts[active_indices[leaf_condition]]
                
                # This indexing is for self.sum_arr indexes
                out_priorities[active_indices[leaf_condition]] = self._cumul_priority[tree_inds[active_indices[leaf_condition]]]
                active[active_indices[leaf_condition]] = False
            
            # Process non-leaf nodes
            non_leaf_condition = ~leaf_condition
            if len(active_indices[non_leaf_condition]) > 0:
                # Get relevant child indices and priorities
                # Determine branch directions
                
                # This operation is valid because left_child_ind is a subset of tree_inds and is exactly the same length as active_indices
                go_left = cumul_targets[active_indices[non_leaf_condition]] < self._cumul_priority[left_childs_ind[non_leaf_condition]]
                go_right = ~go_left
                
                # Process left branches
                if np.any(go_left):
                    # This operation is valid because left_child_ind is a subset of tree_inds and is exactly the same length as active_indices
                    tree_inds[active_indices[non_leaf_condition][go_left]] = left_childs_ind[non_leaf_condition][go_left]
                    
                    #This operation is obviously valid, since the subsetting is  eact same on both sides
                    ends[active_indices[non_leaf_condition][go_left]] = (starts[active_indices[non_leaf_condition][go_left]] +
                                                                        ends[active_indices[non_leaf_condition][go_left]]) // 2
                
                # Process right branches
                if np.any(go_right):
                    # Update targets with precision check
                    
                    # This operation is valid because left_child_ind is a subset of tree_inds and is exactly the same length as active_indices
                    updated_targets = cumul_targets[active_indices[non_leaf_condition]][go_right] - self._cumul_priority[left_childs_ind[non_leaf_condition]][go_right]
                    if np.any(updated_targets < -1e-10):
                        bad_vals = updated_targets[updated_targets < 0]
                        raise ValueError(f"Negative cumul_targets: {bad_vals}. Precision issues.")
                    
                    # Update state for right branch
                    
                    # This operation is valid because the subsetting using active valid and non-right
                    cumul_targets[active_indices[non_leaf_condition][go_right]] = updated_targets
                    
                    # This operation is valid because right_childs_ind is a subset of tree_inds and is exactly the same length as active_indices
                    tree_inds[active_indices[non_leaf_condition][go_right]] = right_childs_ind[non_leaf_condition][go_right]
                    
                    #This operation is obviously valid, since the subsetting is  eact same on both sides
                    starts[active_indices[non_leaf_condition][go_right]] = (
                        starts[active_indices[non_leaf_condition][go_right]] +
                        ends[active_indices[non_leaf_condition][go_right]]
                        ) // 2 + 1

        return out_arr_idxs, out_priorities, [self._arr[i][0] for i in out_arr_idxs]
   
    def sample_arr_index_by_inv_priority(self, cumul_targets: np.ndarray, num_samples):
        """
        Batch‐sample leaf array indices using inverse priorities from the Sum‐Tree.

        Args:
            cumul_targets (np.ndarray, shape=(num_updates,)):
                Values in [0, total_inverse_priority_sum) used to guide each traversal.
            num_samples (int):
                Number of leaves to sample; must equal len(cumul_targets).

        Returns:
            np.ndarray of shape (num_updates,), dtype=int32:
                For each i, the array index in self._arr of the sampled leaf.

        Raises:
            IndexError: If at a leaf, start != end (inconsistent segment).
            ValueError: If updated target becomes < –1e-10 (precision error).
        """
        # Initialize global state arrays
        tree_inds = np.zeros(num_samples, dtype=np.int32)
        starts = np.zeros(num_samples, dtype=np.int32)
        ends = np.full(num_samples, self._capacity - 1, dtype=np.int32)
        active = np.ones(num_samples, dtype=bool)
        result = np.zeros(num_samples, dtype=np.int32)
        

        while np.any(active):
            # Get global indices of active elements
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            
            # Compute child indices for active elements
            left_childs_ind = 2 * tree_inds[active_indices] + 1
            right_childs_ind = 2 * tree_inds[active_indices] + 2
            
            # Identify leaf nodes: out-of-bound OR in-bound with zero priority
            out_of_bound = (left_childs_ind >= self._len_sum_tree)
            leaf_condition = out_of_bound.copy()  # Start with out-of-bound condition
        
            # Check in-bound nodes for zero priority
            if np.any(~out_of_bound):
                # leaf condition for in-bound nodes = inv_prio
                leaf_condition[~out_of_bound] = (self._cumul_inv_priority[left_childs_ind[~out_of_bound]] == 0.0)  # Mark in-bound zero-prio as leaves

            # Process leaf nodes
            if np.any(leaf_condition):
                
                # Validate leaf segments
                bad_mask = starts[active_indices[leaf_condition]] != ends[active_indices[leaf_condition]]
                if np.any(bad_mask):
                    bad_indices = active_indices[leaf_condition][bad_mask]
                    error_msg = "\n".join(
                        f"Index {i}: start={starts[i]}, end={ends[i]}" 
                        for i in bad_indices
                    )
                    raise IndexError(f"Segment mismatch (array len={self._capacity}):\n{error_msg}")
                
                # Store results and deactivate
                result[active_indices[leaf_condition]] = starts[active_indices[leaf_condition]]
                active[active_indices[leaf_condition]] = False
            
            # Process non-leaf nodes
            non_leaf_condition = ~leaf_condition
            if len(active_indices[non_leaf_condition]) > 0:
                # Get relevant child indices and priorities
                # Determine branch directions
                
                # This operation is valid because left_child_ind is a subset of tree_inds and is exactly the same length as active_indices
                go_left = cumul_targets[active_indices[non_leaf_condition]] < self._cumul_inv_priority[left_childs_ind[non_leaf_condition]]
                go_right = ~go_left
                
                # Process left branches
                if np.any(go_left):
                    # This operation is valid because left_child_ind is a subset of tree_inds and is exactly the same length as active_indices
                    tree_inds[active_indices[non_leaf_condition][go_left]] = left_childs_ind[non_leaf_condition][go_left]
                    
                    #This operation is obviously valid, since the subsetting is  eact same on both sides
                    ends[active_indices[non_leaf_condition][go_left]] = (starts[active_indices[non_leaf_condition][go_left]] +
                                                                        ends[active_indices[non_leaf_condition][go_left]]) // 2
                
                # Process right branches
                if np.any(go_right):
                    # Update targets with precision check
                    
                    # This operation is valid because left_child_ind is a subset of tree_inds and is exactly the same length as active_indices
                    updated_targets = cumul_targets[active_indices[non_leaf_condition]][go_right] - self._cumul_inv_priority[left_childs_ind[non_leaf_condition]][go_right]
                    if np.any(updated_targets < -1e-10):
                        bad_vals = updated_targets[updated_targets < 0]
                        raise ValueError(f"Negative cumul_targets: {bad_vals}. Precision issues.")
                    
                    # Update state for right branch
                    
                    # This operation is valid because the subsetting using active valid and non-right
                    cumul_targets[active_indices[non_leaf_condition][go_right]] = updated_targets
                    
                    # This operation is valid because right_childs_ind is a subset of tree_inds and is exactly the same length as active_indices
                    tree_inds[active_indices[non_leaf_condition][go_right]] = right_childs_ind[non_leaf_condition][go_right]
                    
                    #This operation is obviously valid, since the subsetting is  eact same on both sides
                    starts[active_indices[non_leaf_condition][go_right]] = (
                        starts[active_indices[non_leaf_condition][go_right]] +
                        ends[active_indices[non_leaf_condition][go_right]]
                        ) // 2 + 1

        return result

    def update_single_node(self,ind: int, new_td_error: float):
        """
        Updates the TD error of a given index in the sum tree's original array.

        Args:
            ind (int): The index of the node whose priority needs to be updated.
            new_td_error (float): The new TD error value to be assigned.
           
        """
        
        # fetch existing data, leave it as‑is
        old_data, _ = self._arr[ind]
        self.point_update(ind, (old_data, new_td_error))
    
    def update_multiple_nodes(self, indices: List[int] , td_errors: List[float]):
        """
        Updates the td errors of the given list of indices in the sum tree's original array. 
        Both lists must be the same length, not checked here

        Args:
            indices (List[int]) : The indicies in the original array whose td errors and consequently the
                                priorities need to be updated.
            td_errors (List[float]): The td_errors that are to be assigned.
           
        """
        input_tuple_list = []
        
        for i in range(len(indices)):
            # fetch existing data, leave it as‑is
            old_data, _ = self._arr[indices[i]]
            input_tuple_list.append( (old_data, td_errors[i]))
            
        self.range_update(indices, input_tuple_list)

    #Unused function: Probably incorrrect too
    def get_range_sum(self, l:int, r:int):
        """Returns the sum of the range from L to R

        Args:
            l (whole number): left index for sum range
            r (whole number): right index for sum range

        Raises:
            ValueError: if l is greater than r

        Returns:
            Tuple:  (sum from l to r, cumul inverse from l to r)
        """
        
        def _get_range_sum_int(self, l: int, r: int, ql:int, qr:int, tree_ind:int):
            """ Sum is assumed to be inclusive of both indexes l and r

            Args:
                l (whole number): left index for sum range
                r (whole number): right index for sum range
                ql (whole number): left index for query at the current node
                qr (whole number): right index for query at the current node
                tree_ind (int): index of the sum_arr for current node

            Returns:
                Tuple:  (sum from l to r, cumul inverse from l to r)
            """
            if l <= ql and r >= qr:
                return self._cumul_priority[tree_ind], self._cumul_inv_priority[tree_ind], self._min_priority[tree_ind]   # encountered a DataNode

            else:
                mid = (ql + qr) // 2
                priority_sum = 0
                inv_priority_sum = 0

                if l <= mid:  # recurse left
                    (priority, inv_priority) = _get_range_sum_int(l, r, ql, mid, 2 * tree_ind + 1)
                    priority_sum += priority
                    inv_priority_sum+=inv_priority
                if r > mid:
                    (priority, inv_priority) = _get_range_sum_int(l, r, mid + 1, qr, 2 * tree_ind + 2)
                    priority_sum += priority
                    inv_priority_sum+=inv_priority
                return (priority_sum, inv_priority_sum)
        
        if l > r: 
            raise ValueError("left index cannot be greater than right index")

        return _get_range_sum_int(l, r, 0, self._capacity- 1, 0)

# Replacement Policies
def stochastic_priority_replacement(sum_tree: SumTree, input_tuple_list: Tuple[Any, float]):
    """
    Replaces input_tuples at random, weighted by the inverse augmented priority (P_i)**alpha.
    That is, a higher priority index has less of a chance of being kicked out.

    Args:
        sum_tree (SumTree): The SumTree 
        input_tuple_list (List[Tuple[Any, float]]): A list of new (data, td_error) tuples to be added.

    Returns:
        np.NDarray : The priorities (not the td_error) of the original nodes which are replaced.
    """
    num_updates = len(input_tuple_list)
    
    # Randomly select targets in [0, total inverse priority]
    targets = np.random.uniform(0, sum_tree._cumul_inv_priority[0], size=num_updates).astype(dtype=np.float32)

    # Find replacement indices via inverse priority sampling
    replace_indx = sum_tree.sample_arr_index_by_inv_priority(targets, num_updates)
    
    # Store old priorities before replacement
    old_priorities = sum_tree._cumul_priority[sum_tree._arr_to_tree_ind[replace_indx]]
    # Replace the selected entries
    sum_tree.range_update(replace_indx, input_tuple_list)

    return old_priorities

class PrioritizedExperienceReplayBuffer:

    def __init__(self, capacity: int, alpha=0.6, beta=0.7, replacement_policy: Callable[[SumTree, Tuple], float] = stochastic_priority_replacement):
        self.capacity = capacity  # max number of allowed experience
        self.alpha = alpha
        self.beta = beta
        self.replacement_policy = replacement_policy
        self.sum_tree = SumTree(self.capacity, self.alpha)
        self.length = 0  # current number of actual experience
        
    def add_experience(self, input_tuple: Tuple[Any, float]):
        """
        Insert a new experience. Presumes that input input_tuple contains td error.

        Args:
            input_tuple (Tuple[Any, float]): contains experience data and TD Error.


        Returns:
            float: The priority of the replaced DataNode. Will be epsilon if an empty DataNode existed there.
        """
        if self.length < self.capacity:
            self.sum_tree.point_update(self.length, input_tuple)
            self.length += 1
            return self.sum_tree._cumul_priority[
                self.sum_tree._arr_to_tree_ind[self.length -1 ] # prevent an off by one error
                ]
        else:
            return self.replacement_policy(self.sum_tree, [input_tuple])[0]
        
    def add_batch_experience(self, input_tuple_list: List[Tuple[Any, float]]):
        """
        Insert a batch of new (data, td_error) pairs into the replay buffer.

        Args:
            input_tuple_list (List[Tuple[Any, float]]):
                A list of (data, td_error) pairs to insert into the buffer.
                The length of this list may exceed the number of empty slots,
                in which case the surplus will replace existing entries.

        Returns:
            List[float]:
                A list of the new leaf-priority values (cumul_priority) for all
                inserted entries, in the same order as `input_tuple_list`. If
                replacement was used for some tuples, their returned priorities
                come from the replacement policy.

        Raises:
            ValueError:
                If `input_tuple_list` is empty or not a list of tuples of the form
                (data, float).
        """
        if not input_tuple_list or not isinstance(input_tuple_list, list):
            raise ValueError("input_tuple_list must be a non-empty list of (data, td_error) tuples.")

        num_updates = len(input_tuple_list)
        space_left  = self.capacity - self.length

        # Case A: Entire batch fits into the remaining empty slots
        if num_updates <= space_left:
            start_idx = self.length
            end_idx   = self.length + num_updates  # exclusive

            arr_idxs = list(range(start_idx, end_idx))
            self.sum_tree.range_update(arr_idxs, input_tuple_list)

            self.length = end_idx

            # Gather and return the newly assigned priorities
            new_priorities = self.sum_tree._cumul_priority[self.sum_tree._arr_to_tree_ind[arr_idxs]]
            return list(new_priorities)

        # Case B: Batch overflows the remaining empty slots
        else:
            # 1) Fill up empty slots to capacity
            n_to_fill     = space_left
            fill_indices  = list(range(self.length, self.capacity))
            fill_tuples   = input_tuple_list[:n_to_fill]

            if self.length < self.capacity:
                self.sum_tree.range_update(fill_indices, fill_tuples)
            self.length = self.capacity

            # 2) Use replacement policy for the remainder
            replace_tuple_list = input_tuple_list[n_to_fill:]
            # The replacement_policy method should handle picking which existing
            # indices to overwrite and return a list of new priorities for each.
            replaced_priorities = self.replacement_policy(self.sum_tree, replace_tuple_list)

            # 3) Collect priorities from the ‘fill’ phase
            fill_priorities = self.sum_tree._cumul_priority[self.sum_tree._arr_to_tree_ind[fill_indices]]
            # 4) Combine and return all priorities in insertion order
            return list(np.concatenate([fill_priorities , replaced_priorities], axis = 0))
     
    def sample(self, num_samples: int):
        """
        Samples `num_samples` trajectories based on priority.
        
        Returns:
        data:      List[Trajectory]         length B
        weights:   np.ndarray (B,)          importance weights, un‑normalized
        indexes:   np.ndarray (B,)          indices into your sum_tree
        """
        total_priority = self.sum_tree._cumul_priority[0]
        targets = np.random.uniform(low = 0.0, high = total_priority, size = num_samples).astype(dtype=np.float32)
        # although data can be fake, the probability of that is very very low, so no extra steps are taken into account
        indexes, priorities, data = self.sum_tree.sample_data_by_priority(targets, num_samples)
        
        # compute sampling probabilities
        probs = priorities/total_priority

        # importance weights (still need normalization later)
        weights = (self.length * probs) ** (-self.beta)
        weights /=  np.max(weights)

        # return raw data + weights + indexes
        return data, weights, indexes

    def update_leaf_priorities(self,indices: List[int],td_errors: List[float]):
        """
        Updates priorities at specified leaf indices in the sum tree.

        Args:
            indices (List[int]): Leaf indices in the sum tree to update.
            td_errors (List[float]): New TD error values for the corresponding indices.

        Raises:
            ValueError: If lengths of indices, td_errors do not match.
        """
        if len(indices) != len(td_errors):
            raise ValueError(
                f"Mismatch: indices ({len(indices)}) and td_errors ({len(td_errors)}) must have the same length."
            )
        together = set(zip(indices, td_errors))  # deduplicate
        indices, td_errors = zip(*together)      # unzip
        self.sum_tree.update_multiple_nodes(indices=indices, td_errors=td_errors)

    def gather_all(self):
        """Returns all stored input tuples in the buffer."""
        return self.sum_tree._arr[:self.length]
    
    def set_beta(self, beta):
        self.beta = beta
    def get_beta(self):
        return self.beta

    def __str__(self):
        arr = self.gather_all()
        return arr.__str__()
    
    def __iter__(self):
        arr = self.gather_all()
        for tuple in arr:
            yield tuple


    

""" sum_tree usability tests """
# min_td_error = -10
# max_td_error = 10
# num_experiences = 10

# # Generate input tuples instead of InputTuple objects
# input_arr = [(i, random.randint(min_td_error, max_td_error)) for  i in range(num_experiences)]
# sum_tree = DataNodeSumTree(input_arr, alpha=1)

# print(sum_tree.find_min_priority_index(0, 0, len(input_arr)-1))
# print(sum_tree.arr)  # print arr
# print()
# print(sum_tree.sum_arr)

# # Perform point update with input_tuple instead of DataNode
# sum_tree.point_update(0, (4, 9))
# print(sum_tree.sum_arr)
# print(sum_tree.get_range_sum(0, 3))

# # Small test case with explicit input tuples
# sum_tree = DataNodeSumTree([(1, 4), (2, 4), (3, 2)], alpha=1)
# print(sum_tree.find_data_node_index_by_inv_priority(0.2939))




# """ PER useability tests"""

# replaced_priorities_random = []
# replaced_priorities_stochastic = []
# replaced_priorities_min_priority = []
# num_experiments = 10000
# buffer_size = 10 # PER capacity
# num_priorities = 50 # number of times a priority is added to the buffer

# # priorities are randomly generated ints between max and min, both inclusive
# min_td_error = -10
# max_td_error = 10
# num_bins = max_td_error - min_td_error + 1


# ## Tests replacements for different policies and plots their outputs in a histogram

# for _ in range(num_experiments):
#     # Replacement policies
#     per_random = PrioritizedExperienceReplayBuffer(buffer_size, alpha=1, replacement_policy=random_replacement)
#     per_stochastic = PrioritizedExperienceReplayBuffer(buffer_size, alpha=1, replacement_policy=stochastic_priority_replacement)
#     per_min_priority = PrioritizedExperienceReplayBuffer(buffer_size, alpha=1, replacement_policy=min_priority_replacement)

#     for i in range(num_priorities):
#         random_td_error = random.uniform(min_td_error, max_td_error)

#         replaced_priorities_random.append(per_random.add_experience((i, random_td_error)))
#         replaced_priorities_stochastic.append(per_stochastic.add_experience((None, random_td_error)))
#         replaced_priorities_min_priority.append(per_min_priority.add_experience((None, random_td_error)))

# # Plot histograms

# plt.figure(figsize=(12, 5))

# plt.hist(replaced_priorities_random, bins=num_bins, alpha=0.5, label="Random Replacement", color='blue')
# plt.hist(replaced_priorities_stochastic, bins=num_bins, alpha=0.5, label="Stochastic Priority Replacement", color='red')
# plt.hist(replaced_priorities_min_priority, bins=num_bins, alpha=0.5, label="Min Priority Replacement", color='green')

# plt.xlabel("Replaced Priorities")
# plt.ylabel("Frequency")
# plt.title("Comparison of Replacement Policies")
# plt.legend()
# plt.show()

# """PER Tests"""
# # Test parameters
# buffer_size = 5  # PER capacity
# num_experiences = 5  # Number of experiences to add
# num_samples = 3  # Number of samples to draw

# # Initialize PER with stochastic priority replacement
# per = PrioritizedExperienceReplayBuffer(buffer_size, alpha=1, beta=1, replacement_policy=stochastic_priority_replacement)

# # Add experiences with increasing TD errors
# experiences = [(i, i + 1) for i in range(num_experiences)]
# for exp in experiences:
#     per.add_experience(exp)

# print("Initial PER state:")
# print(per)

# # Easy case: Sample from full buffer
# sampled_trajectories, importance_weights, sampled_indexes = per.sample(num_samples)
# print("Sampled Trajectories (Easy Case):", sampled_trajectories)
# print("Importance Weights:", importance_weights)
# print("Sampled Indexes:", sampled_indexes)

# # Easy case: Update priorities using gathered indexes
# new_td_errors = [random.uniform(1, 5) for _ in range(buffer_size)]
# stored_experiences = per.gather_all()
# stored_indexes = list(range(len(stored_experiences)))  # Indices of all stored experiences
# per.update_priorities(stored_indexes, new_td_errors)
# print("PER state after updating priorities (Easy Case):")
# print(per)

# # Edge case: Mismatched input lengths for update_priorities
# try:
#     per.update_priorities(stored_indexes[:3], new_td_errors[:2])  # Different lengths
# except ValueError as e:
#     print(f"Caught expected ValueError: {e}")

# # Edge case: Updating a non-existent index
# try:
#     per.update_priorities([999], [0.5])  # Non-existent index
# except IndexError as e:
#     print(f"Caught expected IndexError: {e}")

# # Interleaved test:
# # 1. Add a new experience.
# # 2. Sample from the buffer.
# # 3. Gather current indexes.
# # 4. Update priorities for these experiences.
# # 5. Sample again.

# per.add_experience((99, 10))
# sampled_trajectories, importance_weights, sampled_indexes = per.sample(num_samples)
# print("Sampled Trajectories after adding a new experience:", sampled_trajectories)
# print("Repeated samples allowed:", len(set(sampled_trajectories)) <= len(sampled_trajectories))  # Check for duplicates

# # Gather valid indexes from the buffer before updating priorities
# updated_indexes = list(range(len(per.gather_all())))  # Indices of current experiences
# new_td_errors = [random.uniform(2, 6) for _ in updated_indexes]
# per.update_priorities(updated_indexes, new_td_errors)
# print("PER state after interleaved update:")
# print(per)

# sampled_trajectories, importance_weights, sampled_indexes = per.sample(num_samples)
# print("Final Sampled Trajectories after updates:", sampled_trajectories)
# print("Final Importance Weights:", importance_weights)
# print("Final Sampled Indexes:", sampled_indexes)

# ### Test: Ensure unique indexes while preserving order before updating priorities
# sampled_trajectories, importance_weights, sampled_indexes = per.sample(num_samples)

# # Use a set to remove duplicates while preserving order
# unique_pairs = list({(t, i) for t, i in zip(sampled_trajectories, sampled_indexes)})

# # Unpack trajectories and indexes
# unique_trajectories, unique_indexes = zip(*unique_pairs)

# # Generate new TD errors only for unique indexes
# new_td_errors = [random.uniform(2, 6) for _ in unique_indexes]

# # Update priorities only for unique indexes
# per.update_priorities(list(unique_indexes), new_td_errors)

# print("Successfully updated priorities for unique indexes while preserving order.")


