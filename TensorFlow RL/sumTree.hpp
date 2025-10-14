#ifndef SUM_TREE_HPP
#define SUM_TREE_HPP

#include <vector>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <deque>
#include <unordered_set>
#include <random>
#include <set>
template<typename T>
class SumTree {
public:
    using DataTuple = std::pair<T, float>;

    int capacity;
    float alpha;
    float epsilon = 1e-5;
    float sum_tree_len;
    std::vector<DataTuple> arr ; // Data Array
    std::vector<int> arr_to_tree_index; // index converter array

    std::vector<float> cumul_priority; // sumtree
    std::vector<float> cumul_inv_priority; // sumtree
    std::vector<float> min_priority; // sumtree
    int arr_min_priority_index;
    

    SumTree(int capacity, float alpha)
        : 
        capacity(capacity),
        alpha(alpha),
        epsilon(1e-5f),
        sum_tree_len(2 * pow(2, ceil(log2(capacity))) - 1),
        arr(capacity),
        arr_to_tree_index(capacity),
        cumul_priority((size_t)(2 * std::pow(2, std::ceil(std::log2(capacity))) - 1), 0.0f),
        cumul_inv_priority((size_t)(2 * std::pow(2, std::ceil(std::log2(capacity))) - 1), 0.0f),
        min_priority((size_t)(2 * std::pow(2, std::ceil(std::log2(capacity))) - 1), std::numeric_limits<float>::max())
        {
            build_sum_tree(0, capacity-1, 0);
            arr_min_priority_index = find_min_priority_index();
            

            // initialize arr_to_tree_index
            for (int arr_ind = 0; arr_ind < capacity; arr_ind++){
                int start = 0, end = capacity-1, tree_ind = 0;

                while(start != end){
                    int mid = (start + end) / 2;
                    if(arr_ind > mid){
                        start = mid + 1;
                        tree_ind = 2 * tree_ind + 2;
                    } else {
                        end = mid;
                        tree_ind = 2 * tree_ind + 1;
                    }
                }
                arr_to_tree_index[arr_ind] = tree_ind;
            }
        };
    
    
    /**
     * @brief Batch-update multiple leaves in the sum tree.
     *
     * Overwrites each leaf's data and td_error, then updates its priority as:
     *   priority = (|td_error| + ε)^α
     * After updating the leaves, the changes are percolated up to the root.
     *
     * @param arr_inds A vector of indices in the data array to update.
     * @param input_tuple_vec A vector of (data, td_error) pairs, one per index in `arr_inds`.
     *
     * @return void
     */
    void range_update(std::vector<int> arr_inds, std::vector<DataTuple> input_tuple_vec) {

        int num_updates = arr_inds.size();
        std::vector<float> priorities(num_updates);

        // 1) Update leaves
        for (size_t i = 0; i < input_tuple_vec.size(); i++) {
            float td_err = input_tuple_vec[i].second;
            priorities[i] = std::pow(std::abs(td_err) + epsilon, alpha);
        }

        // together these 2 structures pretend to be an orderset set
        std::deque<int> fifo_queue;
        std::unordered_set<int> unique_set; 

        int leaf;      // used in the for loop below
        float new_p;

        for (int i = 0; i < num_updates; i++) {
            leaf = arr_to_tree_index[arr_inds[i]]; // convert the array to tree indexes
            if (unique_set.find(leaf) != unique_set.end()) continue;

            // update the data array
            arr[arr_inds[i]] = input_tuple_vec[i];
            
            new_p = priorities[i];
            cumul_priority[leaf]     = new_p;
            cumul_inv_priority[leaf] = 1.0f / new_p;
            min_priority[leaf]       = new_p;

            fifo_queue.push_back(leaf);
            unique_set.insert(leaf);
        }

        // 2) Propagate the changes up the sum tree to the root O(k log(N))
        int curr_tree_ind, sibling_indx, parent_ind;

        while (true) {
            // 2.a) get the new index to process (FIFO)
            curr_tree_ind = fifo_queue.front();
            fifo_queue.pop_front(); // Efficient pop

            // 2.b) If already processed, get another index
            if (unique_set.find(curr_tree_ind) == unique_set.end()) continue;

            // 2.c) Otherwise remove from the set, since processing right now
            unique_set.erase(curr_tree_ind);

            // 2.d) Root has no parent; all nodes processed
            if (curr_tree_ind == 0) break;

            // 2.e) calculate parent and sibling indexes and process the parent
            parent_ind = (curr_tree_ind - 1) / 2;
            sibling_indx = (curr_tree_ind % 2 == 0) ? curr_tree_ind - 1 : curr_tree_ind + 1;

            if (sibling_indx >= sum_tree_len) {
                throw std::logic_error(
                    "Sibling node at index " + std::to_string(sibling_indx) +
                    " does not exist for node " + std::to_string(parent_ind) +
                    ". Tree structure may be inconsistent."
                );
            }

            // make the updates
            cumul_priority[parent_ind]     = cumul_priority[curr_tree_ind] + cumul_priority[sibling_indx];
            cumul_inv_priority[parent_ind] = cumul_inv_priority[curr_tree_ind] + cumul_inv_priority[sibling_indx];
            min_priority[parent_ind]       = std::min(min_priority[curr_tree_ind], min_priority[sibling_indx]);

            // 2.f) remove the sibling from the set, if present
            if (unique_set.find(sibling_indx) != unique_set.end()) unique_set.erase(sibling_indx);

            // 2.g) add parent in both set and queue
            fifo_queue.push_back(parent_ind);
            unique_set.insert(parent_ind);
        }

        // 3) Re-evaluate the min priority index
        arr_min_priority_index = find_min_priority_index();
    }
        

    /**
     * @brief Updates the TD errors of the given list of indices in the sum tree's original array.
     *
     * The sizes of `indices` and `td_errors` must match (not checked internally).
     * Updates the corresponding priorities in the sum tree based on the new TD errors.
     *
     * @param indices A vector of indices in the original data array whose TD errors and priorities need updating.
     * @param td_errors A vector of TD errors to assign, one per index in `indices`.
     *
     * @return void
     */
    void update_multiple_nodes(const std::vector<int>& indices, const std::vector<float>& td_errors) {

        std::vector<DataTuple> input_tuple_list(indices.size());

        for (size_t i = 0; i < indices.size(); i++) {
            input_tuple_list[i] = DataTuple(arr[indices[i]].first, td_errors[i]);
        }

        range_update(indices, input_tuple_list);
    }

    /**
     * @brief Samples a DataNode based on priority and retrieves its index in the original array.
     *
     * This function traverses the sum tree to locate a DataNode whose cumulative priority 
     * range contains the given `cumul_target`. It returns the DataNode and the corresponding 
     * index in the original data array.
     *
     * @param cumul_target A random number between 0 and the total priority sum.
     *
     * @return std::pair<DataNode, int>
     *         - The sampled DataNode from the sum tree.
     *         - The corresponding index in the original data array.
     *
     * @throws std::logic_error If traversal reaches a mismatch where start != end.
     * @throws std::runtime_error If `cumul_target` becomes negative during traversal 
     *         (likely due to floating-point precision issues).
     */
    std::tuple<std::vector<int>, std::vector<float>, std::vector<T>>  sample_data_by_priority(std::vector<float> cumul_targets, int num_samples){
        std::vector<int> indices(num_samples);
        std::vector<float> priorities(num_samples); 
        std::vector<T> node_data(num_samples);

        int tree_ind, start, end, left_child, right_child;
        float left_cumul_target, curr_target;

        for(int ii = 0; ii< num_samples; ii++){

            curr_target = cumul_targets[ii];

            // Traverse the sum tree
            tree_ind = 0;
            start = 0;
            end = capacity -1;
            while(true){
                left_child = 2*tree_ind + 1;
                right_child = left_child + 1;

                //If at a leaf node, return the DataNode and its index
                if ((left_child >= sum_tree_len) || (cumul_priority[left_child] == 0)) {
                    if (start == end){
                        indices[ii] = start;
                        priorities[ii] = cumul_priority[tree_ind];
                        node_data[ii] = arr[start].first;
                        break; // break only the while loop
                    }
                    throw std::logic_error(
                        "Mismatch: start=" + std::to_string(start) +
                        ", end=" + std::to_string(end) +
                        ", capacity=" + std::to_string(capacity)
                    );
                }
                left_cumul_target = cumul_priority[left_child];

                // Move left if the target is within the left subtree
                if(curr_target < left_cumul_target){
                    tree_ind = left_child;
                    end = (start + end) / 2;
                }
                // Otherwise move right
                else{
                    curr_target -= left_cumul_target;
                    if (curr_target < 0) throw std::runtime_error(
                        "cumul_target became negative: " + std::to_string(curr_target) +
                        ". Possible floating-point precision issue."
                    );
                    tree_ind = right_child;
                    start = (start + end) / 2 + 1;
                }
            }
        }
        return {indices, priorities, node_data};
    }


    /**
    * @brief Batch-sample leaf array indices using inverse priorities from the Sum-Tree.
    *
    * This function traverses the inverse-priority sum tree for each input target,
    * locating the leaf whose cumulative inverse-priority range contains the target.
    * The result is the set of indices in the original data array corresponding to
    * the sampled leaves.
    *
    * @param cumul_targets Vector of length num_samples containing values in the range
    *        [0, total_inverse_priority_sum), used to guide each traversal.
    * @param num_samples Number of leaves to sample; must equal cumul_targets.size().
    *
    * @return std::vector<int> A vector of size num_samples, where each entry is the
    *         array index in the original data array corresponding to the sampled leaf.
    *
    * @throws std::logic_error If traversal reaches a leaf but start != end, indicating
    *         an inconsistent tree segment.
    * @throws std::runtime_error If the updated cumulative target becomes negative
    *         (e.g., < –1e-10), likely due to floating-point precision issues.
    */
    std::vector<int> sample_arr_index_by_inv_priority(std::vector<float> cumul_targets, int num_samples){
        std::vector<int> indices(num_samples);

        int tree_ind, start, end, left_child, right_child;
        float left_cumul_target, curr_target;

        for(int ii = 0; ii< num_samples; ii++){

            curr_target = cumul_targets[ii];

            // Traverse the sum tree
            tree_ind = 0;
            start = 0;
            end = capacity -1;
            while(true){
                left_child = 2*tree_ind + 1;
                right_child = left_child + 1;

                //If at a leaf node, return the DataNode and its index
                if ((left_child >= sum_tree_len) || (cumul_inv_priority[left_child] == 0)) {
                    if (start == end){
                        indices[ii] = start;
                        break; // break only the while loop
                    }
                    throw std::logic_error(
                        "Mismatch: start=" + std::to_string(start) +
                        ", end=" + std::to_string(end) +
                        ", capacity=" + std::to_string(capacity)
                    );
                }
                left_cumul_target = cumul_inv_priority[left_child];

                // Move left if the target is within the left subtree
                if(curr_target < left_cumul_target){
                    tree_ind = left_child;
                    end = (start + end) / 2;
                }
                // Otherwise move right
                else{
                    curr_target -= left_cumul_target;
                    if (curr_target < 0) throw std::runtime_error(
                        "cumul_target became negative: " + std::to_string(curr_target) +
                        ". Possible floating-point precision issue."
                    );
                    tree_ind = right_child;
                    start = (start + end) / 2 + 1;
                }
            }
        }
        return indices;
    }



private:
     /**
         * @brief Recursively initializes the sum-tree arrays over the range [start, end].
         *
         * For each leaf node (when start == end), this sets:
         *   - cumul_priority      = ε
         *   - cumul_inv_priority  = 1 / ε
         *   - min_priority        = ε
         *
         * For each internal node, this sets:
         *   - cumul_priority      = left.cumul_priority + right.cumul_priority
         *   - cumul_inv_priority  = left.cumul_inv_priority + right.cumul_inv_priority
         *   - min_priority        = min(left.min_priority, right.min_priority)
         *
         * @param start    Left index of the segment (inclusive).
         * @param end      Right index of the segment (inclusive).
         * @param tree_ind Index in the tree arrays (`cumul_priority`, `cumul_inv_priority`,
         *                 `min_priority`) where this node’s values will be stored.
         *
         * @return std::array<float,3> containing:
         *         [ cumulative_priority, cumulative_inverse_priority, minimum_priority ]
         */
    std::array<float,3>  build_sum_tree(int start, int end, int tree_ind){
        if (start == end){ // leaf node
            cumul_priority[tree_ind] = this->epsilon;
            cumul_inv_priority[tree_ind] = 1.0f/epsilon;
            min_priority[tree_ind] = epsilon;

            return {epsilon, 1.0f/epsilon, epsilon};
        }

        int mid = (start + end)/2; // integer diviion
        
        std::array<float,3>  left_child = build_sum_tree(start, mid, 2*tree_ind + 1);
        std::array<float,3>  right_child = build_sum_tree(mid + 1, end, 2*tree_ind + 2);

        float cumul_p = left_child[0] + right_child[0];
        float cumul_inv_p = left_child[1] + right_child[1];
        float min_p = std::min( left_child[2],right_child[2]);

        cumul_priority[tree_ind] = cumul_p;
        cumul_inv_priority[tree_ind] = cumul_inv_p;
        min_priority[tree_ind] = min_p;

        return {cumul_p, cumul_inv_p, min_p};

    }

    /**
     * @brief Finds the leaf node in the sum tree whose min_priority is minimal.
     *
     * Traverses the tree from the root to locate the leaf node that contains
     * the minimum priority. Returns the index in the original data array
     * corresponding to that leaf.
     *
     * @return int Index in the original data array corresponding to the leaf
     *             with minimal priority.
     *
     * @throws std::logic_error If neither the left nor right child contains
     *                          the parent's min_priority, indicating an
     *                          inconsistent tree state.
     */
    int find_min_priority_index() {
        int start = 0;
        int end = capacity - 1;
        int tree_ind = 0;

        int mid, left_child, right_child;

        while (true) {
            // If we are at a leaf and its min_priority equals its cumul_priority, return index
            if (min_priority[tree_ind] == cumul_priority[tree_ind] && start == end) {
                return start;
            }

            left_child  = 2 * tree_ind + 1;
            right_child = 2 * tree_ind + 2;
            mid = (start + end) / 2;

            // Go left if left child's min_priority matches current node
            if (min_priority[tree_ind] == min_priority[left_child]) {
                tree_ind = left_child;
                end = mid;
            }
            // Otherwise go right if right child's min_priority matches
            else if (min_priority[tree_ind] == min_priority[right_child]) {
                tree_ind = right_child;
                start = mid + 1;
            }
            // Neither child matches → tree is inconsistent
            else {
                throw std::logic_error(
                    "Invalid sum tree structure at index " + std::to_string(tree_ind) +
                    ": min_priority (" + std::to_string(min_priority[tree_ind]) + 
                    ") not found in left (" + std::to_string(min_priority[left_child]) + 
                    ") or right (" + std::to_string(min_priority[right_child]) + ") child."
                );
            }
        }
    }

};

// Static random number generator for efficiency
static std::mt19937& get_random_generator() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return gen;
}

template<typename T>
std::vector<float> stochastic_priority_replacement(SumTree<T>& sum_tree, const std::vector<std::pair<T, float>>& input_tuple_list) {
    int num_updates = static_cast<int>(input_tuple_list.size());
    std::vector<float> old_priorities(num_updates);

    // Get the random generator
    std::mt19937& gen = get_random_generator();
    
    // Create distribution with current total inverse priority
    std::uniform_real_distribution<float> dist(0.0f, sum_tree.cumul_inv_priority[0]);

    // Generate targets
    std::vector<float> targets(num_updates);
    for (int i = 0; i < num_updates; ++i) targets[i] = dist(gen);

    // Find replacement indices
    std::vector<int> replace_indices = sum_tree.sample_arr_index_by_inv_priority(targets, num_updates);

    // Store old priorities before replacement
    for (int i = 0; i < num_updates; ++i) {
        old_priorities[i] = sum_tree.cumul_priority[sum_tree.arr_to_tree_index[replace_indices[i]]];
    }

    // Replace the selected entries
    sum_tree.range_update(replace_indices, input_tuple_list);

    return old_priorities;
}

template<typename T>
class PrioritizedExperienceReplayBuffer {
public:
    using DataTuple = std::pair<T, float>;
    using ReplacementPolicy = std::function<
        std::vector<float>(SumTree<T>&, const std::vector<std::pair<T, float>>&)>;

    int capacity;
    int length;
    float alpha;
    float beta;
    SumTree<T> sumtree;
    ReplacementPolicy replacement_policy;

    PrioritizedExperienceReplayBuffer(int capacity, float alpha = 0.6, float beta = 0.7,
        ReplacementPolicy replacement_policy = [](SumTree<T>& sum_tree, 
                                                 const std::vector<std::pair<T, float>>& input_tuple_list) {
            return stochastic_priority_replacement(sum_tree, input_tuple_list);
        })
        : capacity(capacity),
          length(0),
          alpha(alpha),
          beta(beta),
          sumtree(capacity, alpha),
          replacement_policy(replacement_policy) {};

    
    /**
     * @brief Insert a batch of new (data, td_error) pairs into the replay buffer.
     *
     * This method adds all entries from `input_tuple_list` into the buffer.
     * - If there is enough space at the end of the buffer, entries are appended directly.
     * - If the batch exceeds available space, the surplus entries are inserted using the
     *   `replacement_policy`, which typically replaces existing entries probabilistically.
     *
     * @param input_tuple_list A vector of DataTuple (std::pair<T, float>) representing
     *                         (data, td_error) pairs to insert. The size may exceed the
     *                         number of empty slots in the buffer.
     *
     * @return std::vector<float> A vector of the new leaf priorities (cumul_priority)
     *                            for all inserted entries, in the same order as
     *                            `input_tuple_list`. If replacement was used for some
     *                            tuples, their returned priorities come from the
     *                            replacement policy.
     *
     * @throws std::invalid_argument If `input_tuple_list` is empty.
     * @throws std::range_error If `input_tuple_list` contains more updates than the buffer
     *                          can handle (should be prevented by replacement).
     */
    std::vector<float> add_batch_experience(std::vector<DataTuple> input_tuple_list){
        int num_updates = input_tuple_list.size();

        int space_to_end = capacity - length;
        int in_capacity  = std::min(num_updates, space_to_end);
        int out_capacity = num_updates - in_capacity;

        std::vector<float> new_priorities(num_updates);

        if (in_capacity > 0 ){
            std::vector<int> in_indices(in_capacity);
            std::iota(in_indices.begin(), in_indices.end(), length);

            std::vector<DataTuple> in_tuples(in_capacity);
            for(int i = 0; i < in_capacity; i++) in_tuples[i] = input_tuple_list[i];

            sumtree.range_update(in_indices, in_tuples);
            length += in_capacity;
            for(int i = 0; i < in_capacity; i++) {
                new_priorities[i] = sumtree.cumul_priority[sumtree.arr_to_tree_index[in_indices[i]]];
            }
        }

        if (out_capacity > 0 ){
            std::vector<DataTuple> out_tuples(out_capacity);
            for(int i = 0; i < out_capacity; i++) out_tuples[i] = input_tuple_list[in_capacity + i];

            std::vector<float> out_priorities = replacement_policy(sumtree, out_tuples);
            
            length = capacity;
            for(int i = 0; i < out_capacity; i++) {
                new_priorities[in_capacity + i] = out_priorities[i];
            }
        }
        return new_priorities;
    }

    /**
    * Samples `num_samples` trajectories from the buffer according to their priorities.
    *
    * @param num_samples  Number of trajectories to sample.
    * 
    * @return A tuple containing three elements:
    *   - std::vector<T> data
    *       The sampled trajectories. Length = num_samples.
    *   - std::vector<float> weights
    *       The corresponding importance-sampling weights (un-normalized).
    *   - std::vector<int> indexes
    *       Indices of the sampled trajectories in the sum tree.
    */
    std::tuple<std::vector<T>, std::vector<float>, std::vector<int>> sample(int num_samples){

        float total_priority = sumtree.cumul_priority[0];

        std::vector<float> targets(num_samples);
        std::mt19937& gen = get_random_generator();
        std::uniform_real_distribution<float> dist(0.0f, total_priority);
        std::generate(targets.begin(), targets.end(), [&](){ return dist(gen); });
        
        auto [indexes, priorities, data] = sumtree.sample_data_by_priority(targets, num_samples);

        std::vector<float> weights(num_samples);
        float max_prio = *std::max_element(priorities.begin(), priorities.end());
        float max_weight = pow(length * max_prio / total_priority, -beta);

        for(int i =0; i<num_samples; i++){
            float p = priorities[i]/total_priority;
            weights[i] = pow(length * p, -beta)/max_weight;
        }
        
        return {data, weights, indexes};
    }

    /**
     * Updates priorities at the specified leaf indices in the sum tree.
     *
     * @param indices  Vector of leaf indices in the sum tree to update.
     * @param td_errors  Vector of new TD error values corresponding to each index.
     *
     * @throws std::invalid_argument  If the lengths of `indices` and `td_errors` do not match.
     */
    void update_leaf_priorities(std::vector<int> indices, std::vector<float> td_errors){
        if (indices.size() != td_errors.size())
            throw std::invalid_argument("Mismatch: indices and td_errors must have the same length.");

        // Deduplicate
        std::set<std::pair<int,float>> unique_pairs;
        
        for (size_t i = 0; i < indices.size(); ++i)
            unique_pairs.insert({indices[i], td_errors[i]});

        // Unzip back into separate vectors
        std::vector<int> dedup_indices;
        std::vector<float> dedup_td_errors;
        for (const auto& p : unique_pairs) {
            dedup_indices.push_back(p.first);
            dedup_td_errors.push_back(p.second);
        }
        sumtree.update_multiple_nodes(dedup_indices, dedup_td_errors);
    }

    void set_beta(float new_beta) {
        beta = new_beta;
    }

    float get_beta() const {
        return beta;
    }


};

#endif // SUM_TREE_HPP
