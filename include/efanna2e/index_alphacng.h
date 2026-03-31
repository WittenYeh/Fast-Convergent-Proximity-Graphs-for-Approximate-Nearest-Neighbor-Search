#ifndef EFANNA2E_INDEX_TAUMNG_H
#define EFANNA2E_INDEX_TAUMNG_H

#include "distance.h"
#include "util.h"
#include "parameters.h"
#include "neighbor.h"
#include "index.h"
#include <cassert>
#include <unordered_map>
#include <string>
#include <sstream>
#include <boost/dynamic_bitset.hpp>
#include <stack>
#include <set>
#include <map>
#include <bits/stdc++.h> 


namespace efanna2e {

class IndexAlphaCNG : public Index {
 public:
  explicit IndexAlphaCNG(const size_t dimension, const size_t n, Metric m, Index *initializer);


  virtual ~IndexAlphaCNG();

  virtual void Save(const char *filename)override;
  virtual void Load(const char *filename)override;


  virtual void Build(size_t n, const float *data, const Parameters &parameters) override;




  virtual void Search(
      const float *query,
      const float *x,
      size_t k,
      const Parameters &parameters,
      unsigned *indices) override;
void Search_(const float *query, const float *x, size_t K,
                      const Parameters &parameters, unsigned *indices,
                      int gtNN, unsigned &query_iterations, unsigned &query_computations);


  

  float eval_recall(std::vector<std::vector<unsigned> > query_res, 
        std::vector<std::vector<int> > gts,
        int K);



  protected:
    typedef std::vector<std::vector<unsigned > > CompactGraph;
    typedef std::vector<SimpleNeighbors > LockGraph;
    typedef std::vector<nhood> KNNGraph;


    CompactGraph final_graph_;

    Index *initializer_;
    void init_graph(const Parameters &parameters);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        boost::dynamic_bitset<>& flags,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);
    //void add_cnn(unsigned des, Neighbor p, unsigned range, LockGraph& cut_graph_);
    void InterInsert_new(unsigned n, unsigned range,
      std::vector<std::vector<SimpleNeighbor>> &reverse_buffer,
      std::vector<std::mutex> &locks,
      SimpleNeighbor *cut_graph_);
      void InterInsert(unsigned n, unsigned range,
                           std::vector<std::mutex> &locks,
                           SimpleNeighbor *cut_graph_);
      void PruneReverseEdges(unsigned n, unsigned range,
        std::vector<std::vector<SimpleNeighbor>> &reverse_buffer,SimpleNeighbor *cut_graph_);
    void sync_prune(unsigned q, std::vector<Neighbor>& pool, const Parameters &parameter, boost::dynamic_bitset<>& flags, SimpleNeighbor* cut_graph_);
    void Link(const Parameters &parameters, SimpleNeighbor* cut_graph_);
    void Load_nn_graph(const char *filename);
    void tree_grow(const Parameters &parameter);
    void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);
    void findroot(boost::dynamic_bitset<> &flag, unsigned &root, const Parameters &parameter);


  private:
    unsigned width;
    unsigned ep_;
    std::vector<std::mutex> locks;
    char* opt_graph_;
    size_t node_size;
    size_t data_len;
    size_t neighbor_len;
    KNNGraph nnd_graph;

  public:
    std::vector<std::vector<float> > final_graph_edge_length_;
    // std::vector<std::atomic<unsigned>> reverse_inserts;
    double NDC;
    int hops;
    float alpha;
    float alpha_step;
    float alpha_max;
    float avg_tau;
    std::vector<float> final_alpha_;
    double comp_amount;
    float tau;
    int threshold;
};

}

#endif //EFANNA2E_INDEX_TAUMNG_H




