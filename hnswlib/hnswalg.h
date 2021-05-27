#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>

namespace hnswlib {
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;

    template<typename dist_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t> {
    public:
        static const tableint max_update_element_locks = 65536;
        HierarchicalNSW(SpaceInterface<dist_t> *s) {

        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location, bool nmslib = false, size_t max_elements=0) {
            loadIndex(location, s, max_elements);
        }

        /**
         * @brief Defination of HNSW index.
         *
         * @param s The instance define the distance using in ANN search.
         * @param max_elements 
         * The maximum number of elements could be contained in one HNSW instance.
         * @param M The maximum number of friend-node for each node in HNSW graph.
         * @param ef_construction 
         * The size of dynamical-linked-list which used to hold and update `ef_construction` 
         * nearest nodes which are potential candidated among current node's `M` friend node.
         */
        HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16, size_t ef_construction = 200, size_t random_seed = 100) :
                link_list_locks_(max_elements), link_list_update_locks_(max_update_element_locks), element_levels_(max_elements) {
            max_elements_ = max_elements;

            has_deletions_=false;
            // Gets bits number used by each vector in one element.
            data_size_ = s->get_data_size();
            // Gets the function used to calculate the distance between two vectors.
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            // Ref to @param
            M_ = M;
            maxM_ = M_;
            maxM0_ = M_ * 2;
            // Ref to @param
            ef_construction_ = std::max(ef_construction,M_);
            ef_ = 10;

            // Initialize random-num generators.
            level_generator_.seed(random_seed);
            update_probability_generator_.seed(random_seed + 1);

            /// There are three components for each element:
            ///     1. vector data, the number of bits allocated by it is `data_size_`.
            ///     2. label, which is each element's external-id, the number of bits 
            ///        allocated by it is `sizeof(labeltype)`.
            ///     3. links info, which save the information of this node's friend-nodes, 
            ///        the number of bits allocated by it is `size_links_level0_`.
            ///
            /// So for each element, the bit number it will occupy is `size_links_level0_`, 
            /// which is the sum of its each components memory usage. 
            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
            /// `offsetData_` and `label_offset_` are the memory-address offset about 
            /// the location of vector-data and label relative to each element's address 
            /// start point. These info will be helpful when we copy the element data to 
            /// memory in the order of links info, vector data, label.
            ///
            /// Besides, `offsetLevel0_` is level0-layer's link info, since for each element, 
            /// this data located at first block, so the offset to the start-point of memory 
            /// is zero. 
            offsetData_ = size_links_level0_;
            label_offset_ = size_links_level0_ + data_size_;
            offsetLevel0_ = 0;

            /// The memory that level0 of HNSW index will allocate.
            data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");

            cur_element_count = 0;

            /// The linked-list holds each node which has been visited, distance-calculated, 
            /// and finally eliminated out from potential friend-node list.
            visited_list_pool_ = new VisitedListPool(1, max_elements);



            //initializations for special treatment of the first node
            enterpoint_node_ = -1;
            maxlevel_ = -1;

            /// The detail refer to annotation in `get_linklist` method. 
            linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
            mult_ = 1 / log(1.0 * M_);
            revSize_ = 1.0 / mult_;
        }

        /**
         * @brief
         * Compare with nodes' distance to target node is closer. 
         * Each node is represented as a `std::pair<dist_t, tableint>` instance, while 
         * `dist_t` represent this node's similarity/distance to certain target node, 
         * and `tableint` is this node's internal-id in HNSW index.
         */
        struct CompareByFirst {
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept {
                return a.first < b.first;
            }
        };

        ~HierarchicalNSW() {

            free(data_level0_memory_);
            for (tableint i = 0; i < cur_element_count; i++) {
                if (element_levels_[i] > 0)
                    free(linkLists_[i]);
            }
            free(linkLists_);
            delete visited_list_pool_;
        }

        size_t max_elements_;
        size_t cur_element_count;
        size_t size_data_per_element_;
        size_t size_links_per_element_;

        size_t M_;
        size_t maxM_;
        size_t maxM0_;
        size_t ef_construction_;

        double mult_, revSize_;
        int maxlevel_;


        VisitedListPool *visited_list_pool_;
        std::mutex cur_element_count_guard_;
        
        /// This is a vector of `std::mutex` with each element corresponding one 
        /// level of layer in HNWS index sreucture.
        std::vector<std::mutex> link_list_locks_;

        // Locks to prevent race condition during update/insert of an element at same time.
        // Note: Locks for additions can also be used to prevent this race condition if the querying of KNN is not exposed along with update/inserts i.e multithread insert/update/query in parallel.
        std::vector<std::mutex> link_list_update_locks_;
        tableint enterpoint_node_;

        /// TODO: Adds annotation. 
        size_t size_links_level0_;
        /// Here are some common offset values:
        /// `offsetData_`: 
        ///     This offset refers to the starting of address of each element's memory 
        ///     buffer in HNSW level0 graph info memory buffer, (plus) with this offset,
        ///     we can easily reach the starting bit of each element's vector-data's 
        ///     memory buffer.  
        /// `offsetLevel0_`: 
        ///     Ref to the starting of address of each element's memory buffer in 
        ///     HNSW level0 graph info memory buffer, (plus) with this offset, we can 
        ///     easily reach the starting bit of each element's HNSW-level0-graph info's 
        ///     memory buffer.
        ///     NOTE: 
        ///     Since for each element's memory buffer in `data_level0_memory_`, the info's 
        ///     order is [HNSW-level0-graph info][vector-data][label], so `offsetLevel0_` 
        ///     does not have offset, so, it's offset is zero. 
        size_t offsetData_, offsetLevel0_;

        /// `data_level0_memory_` is a pointer points the starting of memory-buffer of 
        /// all element's basic-info (which includes HNSW-level0-graph linking/connections info, 
        /// vector-data, label).
        char *data_level0_memory_;
        /// This is a c-lang style 2-dim array style memory-buffer, which contains each
        char **linkLists_;
        std::vector<int> element_levels_;

        size_t data_size_;

        bool has_deletions_;


        size_t label_offset_;
        DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_;
        std::unordered_map<labeltype, tableint> label_lookup_;

        std::default_random_engine level_generator_;
        std::default_random_engine update_probability_generator_;

        inline labeltype getExternalLabel(tableint internal_id) const {
            labeltype return_label;
            memcpy(&return_label,(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
            return return_label;
        }

        inline void setExternalLabel(tableint internal_id, labeltype label) const {
            memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
        }

        inline labeltype *getExternalLabeLp(tableint internal_id) const {
            return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
        }

        /**
         * @brief 
         * Gets the memory address of certain element's vector data, which internal id 
         * is `internal_id`.
         *
         * In detail, the address start-point of HNSW level0 is `data_level0_memory_`, so 
         * the address pointing to `internal_id`-th element is 
         * `data_level0_memory_ + internal_id * size_data_per_element_` since each element 
         * has `size_data_per_element_` bits memory usage. And for the bits range occupied by 
         * our target element, the bits offset of vector-data to this memory range's start point 
         * is `offsetData_` bits, so the bit-range for our target updating vactor data is 
         * pointer by `data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_`. 
         */
        inline char *getDataByInternalId(tableint internal_id) const {
            return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }

        int getRandomLevel(double reverse_size) {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(level_generator_)) * reverse_size;
            return (int) r;
        }


        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t lowerBound;
            if (!isMarkedDeleted(ep_id)) {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                top_candidates.emplace(dist, ep_id);
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty()) {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound) {
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;

                std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data;// = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                if (layer == 0) {
                    data = (int*)get_linklist0(curNodeNum);
                } else {
                    data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                }
                size_t size = getListCount((linklistsizeint*)data);
                tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag) continue;
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id))
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }

        mutable std::atomic<long> metric_distance_computations;
        mutable std::atomic<long> metric_hops;

        template <bool has_deletions, bool collect_metrics=false>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef) const {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            dist_t lowerBound;
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;

            while (!candidate_set.empty()) {

                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                if ((-current_node_pair.first) > lowerBound) {
                    break;
                }
                candidate_set.pop();

                tableint current_node_id = current_node_pair.second;
                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
                if(collect_metrics){
                    metric_hops++;
                    metric_distance_computations+=size;
                }

#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                 _MM_HINT_T0);////////////
#endif
                    if (!(visited_array[candidate_id] == visited_array_tag)) {

                        visited_array[candidate_id] = visited_array_tag;

                        char *currObj1 = (getDataByInternalId(candidate_id));
                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                        if (top_candidates.size() < ef || lowerBound > dist) {
                            candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                            _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                         offsetLevel0_,///////////
                                         _MM_HINT_T0);////////////////////////
#endif

                            if (!has_deletions || !isMarkedDeleted(candidate_id))
                                top_candidates.emplace(dist, candidate_id);

                            if (top_candidates.size() > ef)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }

        /**
         * @brief
         * Get top `M` high similarity/distance score element from `top_candidates` while 
         * keeping considering keep high-ways in HNSW graph. See detail in annotations of 
         * `while (queue_closest.size()) {...}`. 
         */
        void getNeighborsByHeuristic2(
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
            if (top_candidates.size() < M) {
                return;
            }

            /// The priority-queue `candidates` in `updatePoint` member function will give 
            /// the high-distance (which means low similiarity score) nodes high priority (
            /// to pop out), since the target is to pop the relative further (less similar) 
            /// nodes out from priority-queue, replace with relatively higher similarity one.
            ///
            /// But at this place, our target is priority poping out the relatively colser/
            /// higher-similarity nodes from `queue_closest` and put it into `return_list` 
            /// until top `M` nearest/highest-similarity nodes has been decided.
            ///
            /// So in this scenario, we let closer-to-target nodes get higher proprity in 
            /// `queue_closest` by using "-1 * distance" as priority score in `queue_closest`, 
            /// so in this way, the higher-similarity/lower-distance and lower-priority nods in 
            /// `top_candidates` can gets a higher priority (to pop out) in `queue_closest`. 
            /// This process is done in following `while (top_candidates.size() > 0) {...}` loop. 
            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            /// `dist_t` for distance type, `tableint` for element's internal-id type.
            std::vector<std::pair<dist_t, tableint>> return_list;
            while (top_candidates.size() > 0) {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }
            /// After above while loop, the `top_candidates` is empty since all elements 
            /// had been poped out. 

            /// The following popping-out iteration of `queue_closest` will helps 
            /// to select out quatify-neighbors of query-node, not only accords distance 
            /// to query-node, but also consider about how to try best keeping high-ways 
            /// among HNSW graph. 
            while (queue_closest.size()) {
                if (return_list.size() >= M)
                    break;
                std::pair<dist_t, tableint> curent_pair = queue_closest.top();
                /// Current poping out node's distance to query-node. 
                dist_t dist_to_query = -curent_pair.first;
                queue_closest.pop();
                /// If this node "good"/quatify to be a neighbor of query-node, 
                /// For now, the "good" means good for HNSW graph to keep more
                /// high graph, see detail in following annotation.
                bool good = true;
                
                /// TODO: Think twice! 
                /// NOTE: Here is an implicit trick to make confusing! 
                ///
                /// Although at each iteration in `while (queue_closest.size()) {...}`, 
                /// we will pop out and get next most closer-to-query nodes which has 
                /// highest '-1 * distance' in current `queue_closest`, but not each 
                /// of these node can be the neighbor of query-node.
                /// 
                /// Besides `M` limitation of `return_list` size, hnswlib sets another 
                /// requirement that for each poping out node at current iteration, it can 
                /// be the neighor of query-node IF AND ONLY IF its distance to all existing 
                /// neighbors in `return_list` is further/larger than tis distance to 
                /// query-node.
                ///
                /// In summary, the above rule tries to not let each node's neighbors be 
                /// too close with each other which will demage the high-way privilege 
                /// of HNSW graph structure.
                /// 
                /// The above idea can ref to:
                ///     * https://mp.weixin.qq.com/s/ynXT945swVVJDFk8UJ-xIQ
                ///     * https://blog.csdn.net/whenever5225/article/details/106863674 
                ///
                /// If not, we believe current popping-out-node is not "good" for HNSW graph 
                /// to keep some high-ways, which means `good = false;`, and we will not 
                /// let this node to be the neighbor of query-node.
                for (std::pair<dist_t, tableint> second_pair : return_list) {
                    dist_t curdist =
                            fstdistfunc_(getDataByInternalId(second_pair.second),
                                         getDataByInternalId(curent_pair.second),
                                         dist_func_param_);;
                    if (curdist < dist_to_query) {
                        good = false;
                        break;
                    }
                }
                if (good) {
                    return_list.push_back(curent_pair);
                }
            }

            for (std::pair<dist_t, tableint> curent_pair : return_list) {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }


        linklistsizeint *get_linklist0(tableint internal_id) const {
            return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        };

        /**
         * @brief
         * Gets the pointer points the start of memory buffer that contains certain id's 
         * element-connections/links info in certain level graph of HNSW index.  
         *
         * @param internal_id
         * The element internal-id that we hope to get the pointer points this element's 
         * level0-graph info memory buffer from `data_level0_memory_`.
         * @param data_level0_memory_
         * It's a pointer points a memory buffer (actually an c-style array). This memory 
         * buffer saves all info of the graph for the layer/level/hierarchical of 0 in HNSW.
         */
        linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
            /// `data_level0_memory_` is the address points the starting of the memory buffer of 
            /// HNSW level0 graph info. 
            /// The `internal_id * size_data_per_element_` is the offset of `internal_id`-th 
            /// element info refer to starting of level0 graph memory buffer.
            /// `offsetLevel0_` is the offset of the memory-buffer of HNSW level0 graph 
            /// connections/links info refer to starting of `internal_id`-th element memory-buffer. 
            return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        };

        /**
         * @brief 
         * Get the pointer points the starting bit of`internal_id`-th element's 
         * `level`-th level/layer graph info.
         * NOTE: 
         * The value of `level` is larger than zero this place. 
         *
         * The reason is, in hnswlib, the HNSW level0 graph info belongs element 'basic info', 
         * which is unified held by `data_level0_memory_` together with element label and 
         * vector-data, and all element have level0 graph info data. 
         *
         * But not every element has graph info with level higher than zero, so higher-level 
         * graph info does not belongs element basic-info and seperately saved in `linkLists_`.
         *
         * The `get_linklist` will extract target memory buffer pointer from `linkLists_`, so 
         * it only supports graph level higher than zero. 
         *
         * @param internal_id Same as in `get_linklist0`.
         * @param level HNSW graph level number/id/order. 
         */
        linklistsizeint *get_linklist(tableint internal_id, int level) const {
            /// `linkLists_` is an `char **`, a c-style 2-dim array. It is mainly used to 
            /// saving the info of HNSW higher levels/layers' graph info for each element.
            /// The higher levels/layers means, levels/layers which number larger than 0.
            /// 
            /// So, `linkLists_[internal_id]` is the pointer points the starting bits for 
            /// `internal_id`-th element's higher-level graph connections info.
            ///
            /// NOTE:
            /// Saying `linkLists_` is an '2-dim array' is not accurate, is actually an 
            /// pointer of an pointer array. Since for each pointer (corresponding to each 
            /// element), the length of the array it points to could be different with each 
            /// other. 
            /// For example, an element only touchs HNSW level0 graph, which corresponding 
            /// pointer in `linkLists_` will points an zero length array. 
            /// 
            /// On the other hand, an element which at most touchs HNSW level_n graph, 
            /// then besides level0 graph info saved in `data_level0_memory_`, its higher 
            /// level graph info is saved in `linkLists_`, with `linkLists_[internal_id]` 
            /// as starting bit, and  
            /// `linkLists_[internal_id] + n * size_links_per_element_` as ending bit.
            ///
            /// So, when we want to get the pointer points `internal_id`-th element's 
            /// `level`-th level/layer graph info, the starting bit of this target memory 
            /// buffer is `linkLists_[internal_id] + (level - 1) * size_links_per_element_`, 
            /// while `size_links_per_element_` represents one level/layer's graph info 
            /// memory usage if that info exists. 
            return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
        };

        /**
         * @brief 
         * Get pointer points `internal_id`-th element's `level`-th level graph connection info. 
         */
        linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
            return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
        };

        tableint mutuallyConnectNewElement(const void *data_point, tableint cur_c,
                                       std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level, bool isUpdate) {
            size_t Mcurmax = level ? maxM_ : maxM0_;
            getNeighborsByHeuristic2(top_candidates, M_);
            if (top_candidates.size() > M_)
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(M_);
            while (top_candidates.size() > 0) {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }

            tableint next_closest_entry_point = selectedNeighbors.back();

            {
                linklistsizeint *ll_cur;
                if (level == 0)
                    ll_cur = get_linklist0(cur_c);
                else
                    ll_cur = get_linklist(cur_c, level);

                if (*ll_cur && !isUpdate) {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                setListCount(ll_cur,selectedNeighbors.size());
                tableint *data = (tableint *) (ll_cur + 1);
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                    if (data[idx] && !isUpdate)
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    data[idx] = selectedNeighbors[idx];

                }
            }

            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {

                std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(selectedNeighbors[idx]);
                else
                    ll_other = get_linklist(selectedNeighbors[idx], level);

                size_t sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                tableint *data = (tableint *) (ll_other + 1);

                bool is_cur_c_present = false;
                if (isUpdate) {
                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        if (data[j] == cur_c) {
                            is_cur_c_present = true;
                            break;
                        }
                    }
                }

                // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
                if (!is_cur_c_present) {
                    if (sz_link_list_other < Mcurmax) {
                        data[sz_link_list_other] = cur_c;
                        setListCount(ll_other, sz_link_list_other + 1);
                    } else {
                        // finding the "weakest" element to replace it with the new one
                        dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                    dist_func_param_);
                        // Heuristic:
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                        candidates.emplace(d_max, cur_c);

                        for (size_t j = 0; j < sz_link_list_other; j++) {
                            candidates.emplace(
                                    fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                 dist_func_param_), data[j]);
                        }

                        getNeighborsByHeuristic2(candidates, Mcurmax);

                        int indx = 0;
                        while (candidates.size() > 0) {
                            data[indx] = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }

                        setListCount(ll_other, indx);
                        // Nearest K:
                        /*int indx = -1;
                        for (int j = 0; j < sz_link_list_other; j++) {
                            dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                            if (d > d_max) {
                                indx = j;
                                d_max = d;
                            }
                        }
                        if (indx >= 0) {
                            data[indx] = cur_c;
                        } */
                    }
                }
            }

            return next_closest_entry_point;
        }

        std::mutex global;
        size_t ef_;

        void setEf(size_t ef) {
            ef_ = ef;
        }


        std::priority_queue<std::pair<dist_t, tableint>> searchKnnInternal(void *query_data, int k) {
            std::priority_queue<std::pair<dist_t, tableint  >> top_candidates;
            if (cur_element_count == 0) return top_candidates;
            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            for (size_t level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    int *data;
                    data = (int *) get_linklist(currObj,level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            if (has_deletions_) {
                std::priority_queue<std::pair<dist_t, tableint  >> top_candidates1=searchBaseLayerST<true>(currObj, query_data,
                                                                                                           ef_);
                top_candidates.swap(top_candidates1);
            }
            else{
                std::priority_queue<std::pair<dist_t, tableint  >> top_candidates1=searchBaseLayerST<false>(currObj, query_data,
                                                                                                            ef_);
                top_candidates.swap(top_candidates1);
            }

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            return top_candidates;
        };

        void resizeIndex(size_t new_max_elements){
            if (new_max_elements<cur_element_count)
                throw std::runtime_error("Cannot resize, max element is less than the current number of elements");


            delete visited_list_pool_;
            visited_list_pool_ = new VisitedListPool(1, new_max_elements);



            element_levels_.resize(new_max_elements);

            std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

            // Reallocate base layer
            char * data_level0_memory_new = (char *) malloc(new_max_elements * size_data_per_element_);
            if (data_level0_memory_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
            memcpy(data_level0_memory_new, data_level0_memory_,cur_element_count * size_data_per_element_);
            free(data_level0_memory_);
            data_level0_memory_=data_level0_memory_new;

            // Reallocate all other layers
            char ** linkLists_new = (char **) malloc(sizeof(void *) * new_max_elements);
            if (linkLists_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
            memcpy(linkLists_new, linkLists_,cur_element_count * sizeof(void *));
            free(linkLists_);
            linkLists_=linkLists_new;

            max_elements_=new_max_elements;

        }

        void saveIndex(const std::string &location) {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, offsetLevel0_);
            writeBinaryPOD(output, max_elements_);
            writeBinaryPOD(output, cur_element_count);
            writeBinaryPOD(output, size_data_per_element_);
            writeBinaryPOD(output, label_offset_);
            writeBinaryPOD(output, offsetData_);
            writeBinaryPOD(output, maxlevel_);
            writeBinaryPOD(output, enterpoint_node_);
            writeBinaryPOD(output, maxM_);

            writeBinaryPOD(output, maxM0_);
            writeBinaryPOD(output, M_);
            writeBinaryPOD(output, mult_);
            writeBinaryPOD(output, ef_construction_);

            output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

            for (size_t i = 0; i < cur_element_count; i++) {
                unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
                writeBinaryPOD(output, linkListSize);
                if (linkListSize)
                    output.write(linkLists_[i], linkListSize);
            }
            output.close();
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i=0) {


            std::ifstream input(location, std::ios::binary);

            if (!input.is_open())
                throw std::runtime_error("Cannot open file");

            // get file size:
            input.seekg(0,input.end);
            std::streampos total_filesize=input.tellg();
            input.seekg(0,input.beg);

            readBinaryPOD(input, offsetLevel0_);
            readBinaryPOD(input, max_elements_);
            readBinaryPOD(input, cur_element_count);

            size_t max_elements=max_elements_i;
            if(max_elements < cur_element_count)
                max_elements = max_elements_;
            max_elements_ = max_elements;
            readBinaryPOD(input, size_data_per_element_);
            readBinaryPOD(input, label_offset_);
            readBinaryPOD(input, offsetData_);
            readBinaryPOD(input, maxlevel_);
            readBinaryPOD(input, enterpoint_node_);

            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, M_);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, ef_construction_);


            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();

            auto pos=input.tellg();


            /// Optional - check if index is ok:

            input.seekg(cur_element_count * size_data_per_element_,input.cur);
            for (size_t i = 0; i < cur_element_count; i++) {
                if(input.tellg() < 0 || input.tellg()>=total_filesize){
                    throw std::runtime_error("Index seems to be corrupted or unsupported");
                }

                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize != 0) {
                    input.seekg(linkListSize,input.cur);
                }
            }

            // throw exception if it either corrupted or old index
            if(input.tellg()!=total_filesize)
                throw std::runtime_error("Index seems to be corrupted or unsupported");

            input.clear();

            /// Optional check end

            input.seekg(pos,input.beg);


            data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
            input.read(data_level0_memory_, cur_element_count * size_data_per_element_);




            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);


            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            std::vector<std::mutex>(max_elements).swap(link_list_locks_);
            std::vector<std::mutex>(max_update_element_locks).swap(link_list_update_locks_);


            visited_list_pool_ = new VisitedListPool(1, max_elements);


            linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
            element_levels_ = std::vector<int>(max_elements);
            revSize_ = 1.0 / mult_;
            ef_ = 10;
            for (size_t i = 0; i < cur_element_count; i++) {
                label_lookup_[getExternalLabel(i)]=i;
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize == 0) {
                    element_levels_[i] = 0;

                    linkLists_[i] = nullptr;
                } else {
                    element_levels_[i] = linkListSize / size_links_per_element_;
                    linkLists_[i] = (char *) malloc(linkListSize);
                    if (linkLists_[i] == nullptr)
                        throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                    input.read(linkLists_[i], linkListSize);
                }
            }

            has_deletions_=false;

            for (size_t i = 0; i < cur_element_count; i++) {
                if(isMarkedDeleted(i))
                    has_deletions_=true;
            }

            input.close();

            return;
        }

        template<typename data_t>
        std::vector<data_t> getDataByLabel(labeltype label)
        {
            tableint label_c;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
                throw std::runtime_error("Label not found");
            }
            label_c = search->second;

            char* data_ptrv = getDataByInternalId(label_c);
            size_t dim = *((size_t *) dist_func_param_);
            std::vector<data_t> data;
            data_t* data_ptr = (data_t*) data_ptrv;
            for (int i = 0; i < dim; i++) {
                data.push_back(*data_ptr);
                data_ptr += 1;
            }
            return data;
        }

        static const unsigned char DELETE_MARK = 0x01;
//        static const unsigned char REUSE_MARK = 0x10;
        /**
         * Marks an element with the given label deleted, does NOT really change the current graph.
         * @param label
         */
        void markDelete(labeltype label)
        {
            has_deletions_=true;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                throw std::runtime_error("Label not found");
            }
            markDeletedInternal(search->second);
        }

        /**
         * Uses the first 8 bits of the memory for the linked list to store the mark,
         * whereas maxM0_ has to be limited to the lower 24 bits, however, still large enough in almost all cases.
         * @param internalId
         */
        void markDeletedInternal(tableint internalId) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
        }

        /**
         * Remove the deleted mark of the node.
         * @param internalId
         */
        void unmarkDeletedInternal(tableint internalId) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur &= ~DELETE_MARK;
        }

        /**
         * Checks the first 8 bits of the memory to see if the element is marked deleted.
         * @param internalId
         * @return
         */
        bool isMarkedDeleted(tableint internalId) const {
            unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId))+2;
            return *ll_cur & DELETE_MARK;
        }

        /**
         * @brief TODO: Figure it out.
         */
        unsigned short int getListCount(linklistsizeint * ptr) const {
            return *((unsigned short int *)ptr);
        }

        /**
         * @brief
         * Sets or resets the size of mem-buffer which saves each node's connection 
         * info at certain layer/level of HNSW index. 
         *
         * @param ptr 
         * A pointer points (the first bit/starting of) an c-style array which saves 
         * one node's neighbors' internal-ids at certain layer of HNSW.
         * @param size 
         * The size/number of this node's neighbors after its connection info have 
         * been updated or created(initialized).
         */
        void setListCount(linklistsizeint * ptr, unsigned short int size) const {
            *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
        }

        void addPoint(const void *data_point, labeltype label) {
            addPoint(data_point, label,-1);
        }

        /**
         * @brief
         * This function has two responsibilities:
         *     1. Based on new vector value on an existed node (which is, `internalId`-th node 
         *        in HNWS index), update all nodes' graph/connection info which used to has 
         *        target node(`internalId`-th node) as its friend-node/neighbor.
         *        Since the vector of target-node has changed, so the connection/graph info 
         *        of all nodes which have connection point to target-node needs to be updated.
         *
         *     2. Update already exist's internal id's corresponding vector data and linking info.
         *        Its linking info will be updated with calling of `repairConnectionsForUpdate`.
         *
         * @param dataPoint Updating vector data.
         * @param internalId Target exists element's internal id.
         * @param updateNeighborProbability TODO:  Figure it out.
         */
        void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
            // update the feature vector associated with existing point with new vector
            /// Two steps:
            ///     1. Get the pointer points the first element of the target-updating vector, 
            ///        done by `getDataByInternalId(internalId)`.
            ///     2. Copy updating data pointed by `dataPoint` to corresponding bits range 
            ///        pointed by getDataByInternalId(internalId)`. 
            memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

            int maxLevelCopy = maxlevel_;
            tableint entryPointCopy = enterpoint_node_;
            // If point to be updated is entry point and graph just contains single element 
            // then just return.
            // Because it is the only point in HNSW and we do not need build any connection.
            if (entryPointCopy == internalId && cur_element_count == 1)
                return;

            /// `element_levels_` saves for each element, the deepest graph layer/level it 
            /// touchs in HNSW index, so when we update an element (data vector), we must also 
            /// update it links info in all layers it involves in. 
            int elemLevel = element_levels_[internalId];
            std::uniform_real_distribution<float> distribution(0.0, 1.0);
            /// The following loop iterates along each level/layer that target updating element 
            /// involves in, and, at that level/layer, updating the graph connection info of 
            /// the node which will be affected by the updating of target element/node. 
            /// 
            /// These "affected" nodes at each composed by the nodes which has target updating 
            /// node as their neighbor/friend-node at this layer/level. For each of these nodes, 
            /// since one of its neighbors/friend-nodes' vector has been updated, so this node's 
            /// graph/connection info at should also be updated.
            for (int layer = 0; layer <= elemLevel; layer++) {
                /// Here are two data-structures used to help recalculation graph/connection 
                /// info of arget-updating existed node's friend nodes, which connection info 
                /// will be affected by the updating of target node's vector. P.S., the target 
                /// updateing node refers to `internalId`-th element's of HNSW index.
                ///
                /// `sCand`: TODO: Make sure twice!
                ///     This used to hold the candidates which have potential to be the new 
                ///     neighbors of the elements which graph/connection info has been affected 
                ///     by `internalId`-th node's vector updating.
                ///
                ///     NOTE: `sCand` has following components: 
                ///     1. `internalId`-th node itself will also in `sCand`, since after its 
                ///        vector has been updated, it also has probibality to be the neighbor 
                ///        of nodes used to be its friend-nodes/neighbors.  
                ///     2. Besides updated `internalId`-th node, `sCand` will also contain the 
                ///        old/historical friend-nodes/neighbors of these updating-affected nodes.
                ///     3. Since all these "updating affected nodes" are neighbors of same node (
                ///        which is `internalId`-th node), so it's possible they are close to each 
                ///        other, so for any one of them, any others could be this one's new 
                ///        neighbor.
                ///     
                ///     Since these "updating-affected nodes" are neighbors/friend-nodes of 
                ///     target-updating-node(`internalId`-th node), so we call them one-hop nodes (
                ///     of target-updating-node), and the friend-nodes/neighbors of these 
                ///     "one-hop nodes" called "two-hop nodes".
                ///            
                /// `sNeigh`: TODO: Make sure twice!
                ///     `sNeigh` saves "one-hop nodes" of target-updating node, which are old 
                ///     friend-nodes/neighbors of it.
                /// 
                ///     Actually, `sNeigh` saves the nodes which may be affected by `internalId`-th 
                ///     node's updating, and, these nodes are `internalId`-th node's neighbors. 
                ///     The graph/connection info of the nodes in `sNeigh` needs update. 
                std::unordered_set<tableint> sCand;
                std::unordered_set<tableint> sNeigh;
                /// Gets `internalId`-th element's `layer`-th level graph connection info, which 
                /// are its "one-hop-node"/friend-node/neighbors, saving there nodes internal-id 
                /// into `listOneHop`. 
                std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
                if (listOneHop.size() == 0)
                    continue;

                /// As mentioned above in annotation of `sCand`, the `internalId`-th node is 
                /// one of the components of `sCand`. 
                sCand.insert(internalId);

                /// The following loop filling the necessary components into `sCand` and `sNeigh`.
                /// But this filling process is not strict, Refer to the "Random Behavior" in 
                /// following annotation.
                for (auto&& elOneHop : listOneHop) {
                    sCand.insert(elOneHop);

                    /// TODO: Think twice.
                    /// Random Behavior 
                    /// Q:
                    /// The following codes looks like involves randomness into candidates 
                    /// selection, but why this only works on `sNeigh` but not also on `sCand`?
                    /// A:
                    /// My guess is, by randomlly discard some potential nodes at this place, we 
                    /// can let the graph keep/left some "high-way" by not let every updated 
                    /// connection points to actual/most-possible nearest nodes. 
                    if (distribution(update_probability_generator_) > updateNeighborProbability)
                        continue;

                    sNeigh.insert(elOneHop);

                    /// Gets "two-hop nodes" mentioned above by extracting the neighbors of 
                    /// "one-hop nodes" (which are the neighbors of target-updating nodes, also 
                    /// the nodes which connection/graph info directly affected by the updating 
                    /// of target node), these "two-hop nodes" are the potential new neighbors 
                    /// of these "one-hop nodes" which graph/connection info waiting update. 
                    std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                    for (auto&& elTwoHop : listTwoHop) {
                        sCand.insert(elTwoHop);
                    }
                }

                /// Iteration along each current stage one-hop candidates. The target for each 
                /// time iteration is, updating this target-updating-node's "one-hop-node"/
                /// neighbor's graph/connection info at HNSW `layer`-th layer, since the 
                /// graph/connection info of this "one-hop-node" of target-updating-node has been 
                /// directly affected for target-node's vector updating, we should recalculate 
                /// its connection info.
                for (auto&& neigh : sNeigh) {
//                    if (neigh == internalId)
//                        continue;
                    
                    /// Here we initialize a new priority-queue `candidates` to ranking the nodes 
                    /// in `sCand`, which are the potential new neighbors of each nodes which 
                    /// connection info is affected be the updating of `internalId`-th element 
                    /// in HNSW index, that is, each element `neigh` in `sNeigh`.
                    ///
                    /// Note, the name 'candidate' may cause confusing, you need remenber that 
                    /// here the 'candidates' means the candidates neighbors for `neigh`.
                    ///
                    /// Since we want update(find new) graph/connections info of `neigh`, so for 
                    /// each element in `sNeigh` (which are old neighbors of `internalId`-th 
                    /// element before its update), we rank all element in `sCand` according its 
                    /// distance/similarity to `neigh`, higher similarity nodes has higher 
                    /// priority, lower similarity nodes has lower similarity, and high priority 
                    /// to wipe them out when considering if this is an neighbor of `neigh`.
                    ///
                    /// Note the priority-queue `candidates` has a size limit decided by following 
                    /// `elementsToKeep`, so priority-queue is continuously/dymical updating by 
                    /// continuously let `candidates` poping out the furtherest node from `neigh` 
                    /// when an closer to `neigh` compare with it has been discovered.
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    /// The following two lines code is sort of getting an adjusted 
                    /// `ef_construction` value.
                    ///
                    /// In HNSW paper, ef_construction is the size of the prior-queue used to 
                    /// dynamical/continuously mining target-nodes' top-M neighbors, the 
                    /// adjusted `ef_construction` will be saved in `elementsToKeep`.
                    size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1; // sCand guaranteed to have size >= 1
                    size_t elementsToKeep = std::min(ef_construction_, size);
                    /// TODO: Make sure twice!
                    /// As mentioned above, what following loop does is:
                    ///     1. Note all `neigh` in `sNeigh` are also contained in `sCand`, the 
                    ///        the reason is each element in `sNeigh` could also be petential 
                    ///        neighbor of each other since all of them belongs to the neighbors 
                    ///        of old target-updating-node, which means they are close to each 
                    ///        other. 
                    ///        With above background, we should avoid consider if `neigh` itself 
                    ///        in `sCand` (certain `cand`) is nerghbor of itself.
                    /// 
                    ///     2. Calculation of the distance bewteen current `neigh` and all its 
                    ///        potential neighbor `cand` in `sCand`, excludes the `cand` same 
                    ///        with `neigh` itself.
                    /// 
                    ///     3. According to adjusted 'ef_construction' `elementsToKeep` to see if 
                    ///        prior-queue `candidates` is saturated, here are several condition:
                    ///        a. `candidates` not saturate:
                    ///           Then any `cand` exclude `cand` same with `neigh` can be inserted 
                    ///           into `candidates` with its distance/similarity to `neigh`.
                    ///
                    ///        b. `candidates` is saturated and current `cand`'s distance/similarity 
                    ///           to `neigh` is closer/larger than the distance/similarity of the 
                    ///           node in `candidates` which has furtherest distance or smallest 
                    ///           similarity among all nodes in `candidates`:
                    ///           Then pop that the node which has furtherest distance or smallest 
                    ///           similarity among all nodes in `candidates` out from `candidates`, 
                    ///           and `emplace` the `cand` into `candidates`.
                    ///
                    ///        c. Similar with condition b, but this time current `cand`'s 
                    ///           distance/similarity does closer/higher than that value of any 
                    ///           nodes in saturated `candidates`:
                    ///           Do nothing about current `cand`, just passed it and go to next 
                    ///           iteration. 
                    for (auto&& cand : sCand) {
                        if (cand == neigh)
                            continue;

                        dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                        if (candidates.size() < elementsToKeep) {
                            candidates.emplace(distance, cand);
                        } else {
                            if (distance < candidates.top().first) {
                                candidates.pop();
                                candidates.emplace(distance, cand);
                            }
                        }
                    }
                    /// After above `for (auto&& cand : sCand) {...}` iteration, for each node 
                    /// which graph/connections info has been affected by the updating of 
                    /// `internalId`-th element, we got an prior-queue which saves most possible 
                    /// candidate neighbors for it. The "most possible" means the candidate 
                    /// neighbors in finally kept by `condidates` are the nodes which has top 
                    /// `elementsToKeep` nearest distance or highest similarity among all 
                    /// potential candidates among `sCand`.

                    /// Retrive neighbours or `neigh` using heuristic and set connections. 
                    /// And doing inplace-update for `candidates` to save the result of updated 
                    /// neighbors of each `neigh`. 
                    /// NOTE:
                    /// Here the retriving rule is not just retriving the nearest/most-optimize 
                    /// neighbors, but also considering about how to keep some high-way in graph.
                    getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                    /// The following code block's job is update HNSW graph structure at 
                    /// `layer`-th level/layer for `neigh` using its new neighbors calculated 
                    /// above, saving in `candidates`.
                    {
                        /// The mutex instance `link_list_locks_[neigh]` is the mutex for
                        /// `neigh`-th node(node which internal-id is `neigh`), so by only 
                        /// lock `link_list_locks_[neigh]`, other thread can still writting 
                        /// to other nodes' related info, this design can improbe concurrency 
                        /// power of the lib.
                        std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                        linklistsizeint *ll_cur;
                        ll_cur = get_linklist_at_level(neigh, layer);
                        size_t candSize = candidates.size();
                        /// Resets the size of mem-buffer of `neigh`-th node at `layer`-th 
                        /// layer/level according the number of its new calculated neighbors 
                        /// saved in `candidates`.
                        setListCount(ll_cur, candSize);
                        /// TODO: Figure out why let pointer `ll_cur` right move a byte?
                        tableint *data = (tableint *) (ll_cur + 1);
                        /// Iteratively gets `neigh`-th node's new-calculated neighbor and 
                        /// records their internal-id into `neigh`-th node's graph-info buffer 
                        /// at `layer`-th layer of HNSW (which starting bit is pointed by `data`), 
                        /// and pop out each of its neighbor from `candidates` and finally clean 
                        /// `candidates` to empty when iteration have been finished. 
                        for (size_t idx = 0; idx < candSize; idx++) {
                            data[idx] = candidates.top().second;
                            candidates.pop();
                        }
                    }
                }
            }

            repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
        };

        void repairConnectionsForUpdate(const void *dataPoint, tableint entryPointInternalId, tableint dataPointInternalId, int dataPointLevel, int maxLevel) {
            tableint currObj = entryPointInternalId;
            if (dataPointLevel < maxLevel) {
                dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxLevel; level > dataPointLevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist_at_level(currObj,level);
                        int size = getListCount(data);
                        tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                        for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                            _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                            tableint cand = datal[i];
                            dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            if (dataPointLevel > maxLevel)
                throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

            for (int level = dataPointLevel; level >= 0; level--) {
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                        currObj, dataPoint, level);

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
                while (topCandidates.size() > 0) {
                    if (topCandidates.top().second != dataPointInternalId)
                        filteredTopCandidates.push(topCandidates.top());

                    topCandidates.pop();
                }

                // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
                // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
                if (filteredTopCandidates.size() > 0) {
                    bool epDeleted = isMarkedDeleted(entryPointInternalId);
                    if (epDeleted) {
                        filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                        if (filteredTopCandidates.size() > ef_construction_)
                            filteredTopCandidates.pop();
                    }

                    currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
                }
            }
        }

        /**
         * @brief 
         * Gets element's connections/links on certain level of layer of HNSW by given an 
         * element's internal id and target level.
         * This includes two steps:
         *     1. Gets `internalId`-th element's `level`-th level graph info mem-buffer 
         *        pointer with `get_linklist_at_level`, assign the result to `data`. 
         *     2. Extracts target connections info and put the result in an `std::vector`, 
         *        each element represents current element's friend-node's internal-id 
         *        at HNSW `level`-th layer graph.
         *
         * @return
         * An `std::vector<tableint>` with each element is one of the neighbors of 
         * `internalId`-th node of HNSW index at `level`-th layer. 
         */
        std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
            std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
            unsigned int *data = get_linklist_at_level(internalId, level);
            int size = getListCount(data);
            std::vector<tableint> result(size);
            tableint *ll = (tableint *) (data + 1);
            memcpy(result.data(), ll,size * sizeof(tableint));
            return result;
        };

        /**
         * @brief 
         * Adds one point/element to HNSW index. Also, a raw point/element is 
         * composed with a vector pointed by `datapoint` and a label `label`. 
         * 
         * We can understand the notion of 'label' in two style, each corresponding 
         * to unique scenario:
         *     1. In searching scenario, we can understand 'label' as an external-id of 
         *        each element/point, for example, user-id, item-id, doc-id, etc.
         *     2. In KNN inference scenario, the 'label' is always the label of each sample, 
         *        the most happened label among nearest elements/points will be KNN 
         *        prediction result. BUT not that easy, since the 'label'-to-internal_id 
         *        is one-one mapping, so we can not just let 'label' be true label, but 
         *        should, for example, concate with external-id.
         *
         * After this element/point has been added into the index, the index will assign 
         * it and internal id, and the mapping info of external-id/label to internal-id  
         * will be saved in `label_lookup_`. 
         * 
         * The assignment rule of internal-id is incremental-addition according the index's 
         * element count. 
         
         * SO, we can also see internal-id as the inserting order of certain 
         * element!
         * 
         * The main use of internal-id is to record if certain element 
         * already exists in HNSW index.
         */
        tableint addPoint(const void *data_point, labeltype label, int level) {

            tableint cur_c = 0;
            {
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.
                std::unique_lock <std::mutex> templock_curr(cur_element_count_guard_);
                // If there is a point/element marked by `label` already exists in index, 
                // then we will reused its historical assigned internal id by update the 
                // element info (e.g., link info) for this internal id's corresponding element.
                auto search = label_lookup_.find(label);
                if (search != label_lookup_.end()) {
                    tableint existingInternalId = search->second;

                    templock_curr.unlock();

                    std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(existingInternalId & (max_update_element_locks - 1))]);
                    updatePoint(data_point, existingInternalId, 1.0);
                    return existingInternalId;
                }

                if (cur_element_count >= max_elements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                };

                cur_c = cur_element_count;
                cur_element_count++;
                label_lookup_[label] = cur_c;
            }

            // Take update lock to prevent race conditions on an element with insertion/update at the same time.
            std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);
            std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
            int curlevel = getRandomLevel(mult_);
            if (level > 0)
                curlevel = level;

            element_levels_[cur_c] = curlevel;


            std::unique_lock <std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;


            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

            // Initialisation of the data and label
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);


            if (curlevel) {
                linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
                if (linkLists_[cur_c] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }

            if ((signed)currObj != -1) {

                if (curlevel < maxlevelcopy) {

                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                    for (int level = maxlevelcopy; level > curlevel; level--) {


                        bool changed = true;
                        while (changed) {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                            data = get_linklist(currObj,level);
                            int size = getListCount(data);

                            tableint *datal = (tableint *) (data + 1);
                            for (int i = 0; i < size; i++) {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
                                dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                if (d < curdist) {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                bool epDeleted = isMarkedDeleted(enterpoint_copy);
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                    if (level > maxlevelcopy || level < 0)  // possible?
                        throw std::runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                            currObj, data_point, level);
                    if (epDeleted) {
                        top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }
                    currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
                }


            } else {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;

            }

            //Releasing lock for the maximum level
            if (curlevel > maxlevelcopy) {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            return cur_c;
        };

        std::priority_queue<std::pair<dist_t, labeltype >>
        searchKnn(const void *query_data, size_t k) const {
            std::priority_queue<std::pair<dist_t, labeltype >> result;
            if (cur_element_count == 0) return result;

            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            for (int level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations+=size;

                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            if (has_deletions_) {
                top_candidates=searchBaseLayerST<true,true>(
                        currObj, query_data, std::max(ef_, k));
            }
            else{
                top_candidates=searchBaseLayerST<false,true>(
                        currObj, query_data, std::max(ef_, k));
            }

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            while (top_candidates.size() > 0) {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }
            return result;
        };

        void checkIntegrity(){
            int connections_checked=0;
            std::vector <int > inbound_connections_num(cur_element_count,0);
            for(int i = 0;i < cur_element_count; i++){
                for(int l = 0;l <= element_levels_[i]; l++){
                    linklistsizeint *ll_cur = get_linklist_at_level(i,l);
                    int size = getListCount(ll_cur);
                    tableint *data = (tableint *) (ll_cur + 1);
                    std::unordered_set<tableint> s;
                    for (int j=0; j<size; j++){
                        assert(data[j] > 0);
                        assert(data[j] < cur_element_count);
                        assert (data[j] != i);
                        inbound_connections_num[data[j]]++;
                        s.insert(data[j]);
                        connections_checked++;

                    }
                    assert(s.size() == size);
                }
            }
            if(cur_element_count > 1){
                int min1=inbound_connections_num[0], max1=inbound_connections_num[0];
                for(int i=0; i < cur_element_count; i++){
                    assert(inbound_connections_num[i] > 0);
                    min1=std::min(inbound_connections_num[i],min1);
                    max1=std::max(inbound_connections_num[i],max1);
                }
                std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
            }
            std::cout << "integrity ok, checked " << connections_checked << " connections\n";

        }

    };

}
