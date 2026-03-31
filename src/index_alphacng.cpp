#include <efanna2e/index_alphacng.h>

#include <omp.h>
#include <bitset>
#include <chrono>
#include <cmath>
#include <boost/dynamic_bitset.hpp>
#include <efanna2e/parameters.h>
#include <set>
#include <bits/stdc++.h> 


namespace efanna2e {
#define _CONTROL_NUM 100


IndexAlphaCNG::IndexAlphaCNG(const size_t dimension, const size_t n, Metric m,
                   Index *initializer)
    : Index(dimension, n, m), initializer_{initializer} {}

IndexAlphaCNG::~IndexAlphaCNG() {}


void IndexAlphaCNG::Save(const char *filename) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  assert(final_graph_.size() == nd_);

  out.write((char *)&width, sizeof(unsigned));
  out.write((char *)&ep_, sizeof(unsigned));
  for (unsigned i = 0; i < nd_; i++) {
    unsigned GK = (unsigned)final_graph_[i].size();
    out.write((char *)&GK, sizeof(unsigned));
    out.write((char *)final_graph_[i].data(), GK * sizeof(unsigned));
  }

  out.close();

  int edge_num = 0;
  for (int i = 0; i < final_graph_.size(); i++) {
    edge_num += final_graph_[i].size();
  }
}



void IndexAlphaCNG::Load(const char *filename) {
  std::ifstream in(filename, std::ios::binary);
  in.read((char *)&width, sizeof(unsigned));
  in.read((char *)&ep_, sizeof(unsigned));

  unsigned cc = 0;
  int edge_num = 0;
  while (!in.eof()) {
    unsigned k;
    in.read((char *)&k, sizeof(unsigned));
    if (in.eof()) break;
    cc += k;
    std::vector<unsigned> tmp(k);
    in.read((char *)tmp.data(), k * sizeof(unsigned));
    edge_num += tmp.size();
    final_graph_.push_back(tmp);
  }
  cc /= nd_;
}


void IndexAlphaCNG::Load_nn_graph(const char *filename) {
  std::ifstream in(filename, std::ios::binary);
  unsigned k;
  in.read((char *)&k, sizeof(unsigned));
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  size_t num = (unsigned)(fsize / (k + 1) / 4);
  in.seekg(0, std::ios::beg);

  final_graph_.resize(num);
  final_graph_.reserve(num);
  unsigned kk = (k + 3) / 4 * 4;
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    final_graph_[i].resize(k);
    final_graph_[i].reserve(kk);
    in.read((char *)final_graph_[i].data(), k * sizeof(unsigned));
  }
  in.close();

  int edge_num = 0;
  for (int i = 0; i < final_graph_.size(); i++) {
    edge_num += final_graph_[i].size();
  }
}



void IndexAlphaCNG::get_neighbors(const float *query, const Parameters &parameter,
                             std::vector<Neighbor> &retset,
                             std::vector<Neighbor> &fullset) {
  unsigned L = parameter.Get<unsigned>("L");
  retset.resize(L + 1);
  std::vector<unsigned> init_ids(L);

  boost::dynamic_bitset<> flags{nd_, 0};
  L = 0;
  for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
    init_ids[i] = final_graph_[ep_][i];
    flags[init_ids[i]] = true;
    L++;
  }
  while (L < init_ids.size()) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    init_ids[L] = id;
    L++;
    flags[id] = true;
  }

  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                    (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    L++;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;

        float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                        (unsigned)dimension_);
        Neighbor nn(id, dist, true);
        fullset.push_back(nn);
        if (dist >= retset[L - 1].distance) continue;
        int r = InsertIntoPool(retset.data(), L, nn); //r is the pos of nn in L. L is sorted. If insertion fail, r = L+1

        if (L + 1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k) 
      k = nk;
    else
      ++k;
  }
}

void IndexAlphaCNG::get_neighbors(const float *query, const Parameters &parameter,
                             boost::dynamic_bitset<> &flags,
                             std::vector<Neighbor> &retset,
                             std::vector<Neighbor> &fullset) {
  unsigned L = parameter.Get<unsigned>("L");

  retset.resize(L + 1);
  std::vector<unsigned> init_ids(L);

  L = 0;
  for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
    init_ids[i] = final_graph_[ep_][i];
    flags[init_ids[i]] = true;
    L++;
  }
  while (L < init_ids.size()) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    init_ids[L] = id;
    L++;
    flags[id] = true;
  }

  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                    (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    fullset.push_back(retset[i]);
    L++;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;

        float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                        (unsigned)dimension_);
        Neighbor nn(id, dist, true);
        fullset.push_back(nn);
        if (dist >= retset[L - 1].distance) continue;
        int r = InsertIntoPool(retset.data(), L, nn);

        if (L + 1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
}



void IndexAlphaCNG::init_graph(const Parameters &parameters) {
  float *center = new float[dimension_];
  for (unsigned j = 0; j < dimension_; j++) center[j] = 0;
  for (unsigned i = 0; i < nd_; i++) {
    for (unsigned j = 0; j < dimension_; j++) {
      center[j] += data_[i * dimension_ + j];
    }
  }
  for (unsigned j = 0; j < dimension_; j++) {
    center[j] /= nd_;
  }
  std::vector<Neighbor> tmp, pool;
  ep_ = rand() % nd_;  
  get_neighbors(center, parameters, tmp, pool);
  ep_ = tmp[0].id; 
}


void IndexAlphaCNG::sync_prune(unsigned q, std::vector<Neighbor> &pool,
  const Parameters &parameter,
  boost::dynamic_bitset<> &flags,
  SimpleNeighbor *cut_graph_) {
unsigned range = parameter.Get<unsigned>("R");
unsigned maxc = parameter.Get<unsigned>("C");
unsigned min_degree=threshold;
float alpha_increment = alpha_step;  
float max_alpha=alpha_max;
width = range;
float current_alpha = alpha;

// size_t distance_computations = 0;
// size_t distance_cache_hit = 0;
// size_t distance_cache_miss = 0;

unsigned start = 0;
std::vector<float> dist_cache(maxc * maxc, -1.0f);  // pairwise distance cache 
std::vector<int> result_pool_indices;                       // index in result

for (unsigned nn = 0; nn < final_graph_[q].size(); nn++) {
unsigned id = final_graph_[q][nn];
if (flags[id]) continue;
float dist =
distance_->compare(data_ + dimension_ * (size_t)q,
   data_ + dimension_ * (size_t)id, (unsigned)dimension_);
pool.push_back(Neighbor(id, dist, true));
}

std::sort(pool.begin(), pool.end());
std::vector<Neighbor> result;

if (pool[start].id == q) start++;
result.push_back(pool[start]);
result_pool_indices.push_back(start);
  while (result.size() < range && ++start < pool.size() && start < maxc) {
    auto &p = pool[start];
    bool occlude = false;

    for (size_t t = 0; t < result.size(); ++t) {
      if (p.id == result[t].id) {
        occlude = true;
        break;
      }

      int i = result_pool_indices[t];
      int j = start;
      if (i > j) std::swap(i, j);
      size_t idx = i * maxc + j;

      float djk = dist_cache[idx];
      if (djk < 0) {
        // ++distance_computations;
        // ++distance_cache_miss;
        djk = distance_->compare(data_ + dimension_ * result[t].id,
                                data_ + dimension_ * p.id,
                                dimension_);
        dist_cache[idx] = djk;
      }else{
        // ++distance_cache_hit;
        // ++distance_computations;
      }

      if (p.distance > current_alpha * djk + (current_alpha + 1) * tau) {
        occlude = true;
        break;
      }
    }

    if (!occlude) {
      result.push_back(p);
      result_pool_indices.push_back(start);
    }
  }

  // Step 3: 动态提高 alpha（若度数不够）
  while (result.size() < min_degree && current_alpha < max_alpha) {
    current_alpha += alpha_increment;
    result.clear();
    start = 0;
    if (pool[start].id == q) start++;
    result.push_back(pool[start]);
    result_pool_indices.resize(1);
    result_pool_indices[0] = start;
    while (result.size() < range && ++start < pool.size() && start < maxc) {
      auto &p = pool[start];
      bool occlude = false;
      for (size_t t = 0; t < result.size(); ++t) {
        if (p.id == result[t].id) {
          occlude = true;
          break;
        }

        int i = result_pool_indices[t];
        int j = start;
        if (i > j) std::swap(i, j);
        size_t idx = i * maxc + j;

        float djk = dist_cache[idx];
        if (djk < 0) {
          // ++distance_computations;
          // ++distance_cache_miss;
          djk = distance_->compare(data_ + dimension_ * result[t].id,
                                  data_ + dimension_ * p.id,
                                  dimension_);
          dist_cache[idx] = djk;
        }else{
          // ++distance_computations;
          // ++distance_cache_hit;
        }

        if (p.distance > current_alpha * djk + (current_alpha + 1) * tau) {
          occlude = true;
          break;
        }
      }

      if (!occlude) {
        result.push_back(p);
        result_pool_indices.push_back(start);
      }
    }
  }

  SimpleNeighbor *des_pool = cut_graph_ + (size_t)q * (size_t)range;
  for (size_t t = 0; t < result.size(); ++t) {
    des_pool[t].id = result[t].id;
    des_pool[t].distance = result[t].distance;
  }
  if (result.size() < range) {
    des_pool[result.size()].distance = -1;
  }
  final_alpha_[q] = current_alpha;
  if (q % 10000 == 0) {
    std::cout << "[DEBUG] q = " << q
              << ", final alpha = " << current_alpha
              << ", result.size() = " << result.size()<< "\n";
              // << ", distance computations = " << distance_computations
              // << ", cache hit = " << distance_cache_hit
              // << ", cache miss = " << distance_cache_miss
              // << ", hit ratio = " 
              // << (distance_cache_hit * 100.0 / (distance_cache_hit + distance_cache_miss + 1e-9)) << "%\n";
  }
}


void IndexAlphaCNG::InterInsert(unsigned n, unsigned range,
                           std::vector<std::mutex> &locks,
                           SimpleNeighbor *cut_graph_) {
  // static size_t interinsert_distance_computations = 0;
  unsigned min_degree=threshold;
  SimpleNeighbor *src_pool = cut_graph_ + (size_t)n * (size_t)range;
  for (size_t i = 0; i < range; i++) {
    if (src_pool[i].distance == -1) break;

    SimpleNeighbor sn(n, src_pool[i].distance);
    size_t des = src_pool[i].id;
    SimpleNeighbor *des_pool = cut_graph_ + des * (size_t)range;

    std::vector<SimpleNeighbor> temp_pool;
    int dup = 0;
    {
      LockGuard guard(locks[des]);
      for (size_t j = 0; j < range; j++) {
        if (des_pool[j].distance == -1) break;
        if (n == des_pool[j].id) {
          dup = 1;
          break;
        }
        temp_pool.push_back(des_pool[j]);
      }
    }
    if (dup) continue;

    temp_pool.push_back(sn);
    if (temp_pool.size() > range) {
      std::vector<SimpleNeighbor> result;
      unsigned start = 0;
      float current_alpha = alpha;  // 使用初始 alpha
      float alpha_increment = 0.1; // 增量
      float max_alpha=1.6;
      std::sort(temp_pool.begin(), temp_pool.end());
      result.push_back(temp_pool[start]);
      float th1=(current_alpha+1)*tau;
      while (result.size() < range && (++start) < temp_pool.size()) {
        auto &p = temp_pool[start];
        bool occlude = false;
        // if (p.distance < th1) {
        //   // 如果当前点距离小于 (alpha + 1) * tau，则直接加入结果集
        //   result.push_back(p);
        //   continue;
        // }
        float th2=(current_alpha-1)*p.distance/current_alpha+((current_alpha+1)/current_alpha)*tau;
        for (unsigned t = 0; t < result.size(); t++) {
          if (p.id == result[t].id) {
            occlude = true;
            break;
          }
          // if (result[t].distance < th1) {
          //   // 如果当前点距离小于 (alpha + 1) * tau，则直接加入结果集
          //   continue;
          // }
          if (result[t].distance < th2) {
            continue;
          }
          float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id,
                                         data_ + dimension_ * (size_t)p.id,
                                         (unsigned)dimension_);
          // interinsert_distance_computations++;
          float dist_existNeigh_p = djk;
          float dist_q_p = p.distance;
          if (dist_q_p > current_alpha * dist_existNeigh_p + (current_alpha + 1) * tau) {
            occlude = true;
            break;
          }
        }
        if (!occlude) result.push_back(p);
      }
       // 检查是否需要调整 alpha
      while (result.size() < range && current_alpha < max_alpha) {
        current_alpha += alpha_increment;  // 增加 alpha
        result.clear();                    // 清空结果
        start = 0;                         // 重置开始位置

        // 再次筛选一次
        result.push_back(temp_pool[start]);
        float th1=(current_alpha + 1) * tau;
        while (result.size() < range && (++start) < temp_pool.size()) {
          auto &p = temp_pool[start];
          bool occlude = false;
          // if (p.distance < th1) {
          //   // 如果当前点距离小于 (alpha + 1) * tau，则直接加入结果集
          //   result.push_back(p);
          //   continue;
          // }
          float th2=(current_alpha-1)*p.distance/current_alpha+((current_alpha+1)/current_alpha)*tau;
          for (unsigned t = 0; t < result.size(); t++) {
            if (p.id == result[t].id) {
              occlude = true;
              break;
            }
            // if (result[t].distance < th1) {
            //   // 如果当前点距离小于 (alpha + 1) * tau，则直接加入结果集
            //   continue;
            // }
            if  (result[t].distance < th2) {
              continue;
            }
            float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id,
                                           data_ + dimension_ * (size_t)p.id,
                                           (unsigned)dimension_);
            // interinsert_distance_computations++;
            float dist_existNeigh_p = djk;
            float dist_q_p = p.distance;
            if (dist_q_p > current_alpha * dist_existNeigh_p + (current_alpha + 1) * tau) {
              occlude = true;
              break;
            }
          }
          if (!occlude) result.push_back(p);
        }
      }
      {
        LockGuard guard(locks[des]);
        for (unsigned t = 0; t < result.size(); t++) {
          des_pool[t] = result[t];
        }
      }
    } else {
      LockGuard guard(locks[des]);
      for (unsigned t = 0; t < range; t++) {
        if (des_pool[t].distance == -1) {
          des_pool[t] = sn;
          if (t + 1 < range) des_pool[t + 1].distance = -1;
          break;
        }
      }
    }
  }
}
void IndexAlphaCNG::InterInsert_new(unsigned n, unsigned range,
  std::vector<std::vector<SimpleNeighbor>> &reverse_buffer,
  std::vector<std::mutex> &locks, SimpleNeighbor *cut_graph_) {

  SimpleNeighbor *src_pool = cut_graph_ + size_t(n) * range;

  for (size_t i = 0; i < range; ++i) {
    if (src_pool[i].distance == -1) break;

    unsigned des = src_pool[i].id;
    float dist = src_pool[i].distance;
    SimpleNeighbor sn(n, dist);

    {
      LockGuard guard(locks[des]);
      reverse_buffer[des].emplace_back(sn);
    }
  }
}
void IndexAlphaCNG::PruneReverseEdges(unsigned n, unsigned range,
  std::vector<std::vector<SimpleNeighbor>> &reverse_buffer,SimpleNeighbor *cut_graph_) {
    std::vector<SimpleNeighbor> temp_pool;
    SimpleNeighbor *src_pool = cut_graph_ + (size_t)n * (size_t)range;
    for (size_t i = 0; i < range; i++) {
    if (src_pool[i].distance == -1) break;
    temp_pool.push_back(src_pool[i]);
    }
    const auto& reverse_edges = reverse_buffer[n];
    temp_pool.insert(temp_pool.end(), reverse_edges.begin(), reverse_edges.end());
    if (temp_pool.size() > range) {
    std::vector<SimpleNeighbor> result;
    std::vector<int> result_pool_indices;
    std::vector<float> dist_cache(temp_pool.size() * temp_pool.size(), -1.0f);
    
    unsigned start = 0;
    float current_alpha = alpha;  // init alpha
    float alpha_increment = alpha_step; // alpha step
    float max_alpha=alpha_max;
    std::sort(temp_pool.begin(), temp_pool.end());
    result.push_back(temp_pool[start]);
    result_pool_indices.push_back(start);
    while (result.size() < range && (++start) < temp_pool.size()) {
    auto &p = temp_pool[start];
    bool occlude = false;

    for (size_t t = 0; t < result.size(); ++t) {
      if (p.id == result[t].id) {
        occlude = true;
        break;
        }
   
      int i = result_pool_indices[t];
      int j = start;
      if (i > j) std::swap(i, j);
      size_t idx = i * temp_pool.size() + j;
    
      float djk = dist_cache[idx];
      if (djk < 0) {
        djk = distance_->compare(data_ + dimension_ * result[t].id,
                                data_ + dimension_ * p.id,
                                dimension_);
        dist_cache[idx] = djk;
      }
    
      if (p.distance > current_alpha * djk + (current_alpha + 1) * tau) {
        occlude = true;
        break;
      }
    }
    if (!occlude) {
      result.push_back(p);
      result_pool_indices.push_back(start);
    }
    }
    
    while (result.size() < range && current_alpha < max_alpha) {
    current_alpha += alpha_increment;  
    result.clear();                    
    result_pool_indices.clear();
    
    start = 0;                         
    
    result.push_back(temp_pool[start]);
    result_pool_indices.push_back(start);
    while (result.size() < range && (++start) < temp_pool.size()) {
    auto &p = temp_pool[start];
    bool occlude = false;

    for (size_t t = 0; t < result.size(); ++t) {
      if (p.id == result[t].id) {
        occlude = true;
        break;
        }
  
      int i = result_pool_indices[t];
      int j = start;
      if (i > j) std::swap(i, j);
      size_t idx = i * temp_pool.size() + j;
    
      float djk = dist_cache[idx];
      if (djk < 0) {
        djk = distance_->compare(data_ + dimension_ * result[t].id,
                                data_ + dimension_ * p.id,
                                dimension_);
        dist_cache[idx] = djk;
      }
    
      if (p.distance > current_alpha * djk + (current_alpha + 1) * tau) {
        occlude = true;
        break;
      }
    }
    if (!occlude) {
      result.push_back(p);
      result_pool_indices.push_back(start);
    }
    }
    }
    {
    unsigned limit = std::min((unsigned)result.size(), range);
    for (unsigned t = 0; t < limit; t++) {
    src_pool[t] = result[t];
    }
    if (limit < range)
    src_pool[limit].distance = -1;
    }
    } else {
    unsigned limit = std::min((unsigned)temp_pool.size(), range);

    for (unsigned t = 0; t < limit; t++) {
    src_pool[t] = temp_pool[t];
    }
    if (limit < range)
      src_pool[limit].distance = -1;
    }
}

void IndexAlphaCNG::Link(const Parameters &parameters, SimpleNeighbor *cut_graph_) {
  
  unsigned range = parameters.Get<unsigned>("R");
  std::vector<std::mutex> locks(nd_);
  final_alpha_.resize(nd_, -1.0f); 
  double start_time1, end_time1, start_time2, end_time2;

  std::vector<std::vector<SimpleNeighbor>> reverse_buffer(nd_);


  start_time1 = omp_get_wtime();
    double total_time_neighbors = 0.0;
  double total_time_prune = 0.0;
#pragma omp parallel
  {
    std::vector<Neighbor> pool, tmp;
    boost::dynamic_bitset<> flags{nd_, 0};


#pragma omp for schedule(dynamic, 100)reduction(+:total_time_neighbors, total_time_prune)
for (unsigned n = 0; n < nd_; ++n) {
  pool.clear();
  tmp.clear();
  flags.reset();

  double start_time_neighbors = omp_get_wtime();
  get_neighbors(data_ + dimension_ * n, parameters, flags, tmp, pool);
  double end_time_neighbors = omp_get_wtime();
  
  double start_time_prune = omp_get_wtime();
  sync_prune(n, pool, parameters, flags, cut_graph_);
  double end_time_prune = omp_get_wtime();

  // 累加总时间
  total_time_neighbors += (end_time_neighbors - start_time_neighbors);
  total_time_prune += (end_time_prune - start_time_prune);
}
  

  }
end_time1 = omp_get_wtime();
std::cout << "Total get_neighbors execution time: " << total_time_neighbors << " seconds" << std::endl;
std::cout << "Total sync_prune execution time: " << total_time_prune << " seconds" << std::endl;
std::cout << "Prune execution time: " << (end_time1 - start_time1) << " seconds" << std::endl;

  start_time2 = omp_get_wtime();
  #pragma omp parallel
    {
  #pragma omp for schedule(dynamic, 100)
      for (unsigned n = 0; n < nd_; ++n) {
        InterInsert(n, range, locks, cut_graph_); 
      }
    }

  end_time2 = omp_get_wtime();
  std::cout << "InterInsert execution time: " << (end_time2 - start_time2) << " seconds" << std::endl;
}

// #pragma omp parallel
//   {
// #pragma omp for schedule(dynamic, 100)
//     for (unsigned n = 0; n < nd_; ++n) {
//       InterInsert(n, range, reverse_buffer,locks,cut_graph_); 
//     }
//   }

//   #pragma omp parallel
//   {
// #pragma omp for schedule(dynamic, 100)
//     for (unsigned n = 0; n < nd_; ++n) {
//       PruneReverseEdges(n, range, reverse_buffer,cut_graph_); 
//     }
//   }

//   end_time2 = omp_get_wtime();
//   std::cout << "InterInsert execution time: " << (end_time2 - start_time2) << " seconds" << std::endl;

// }

void IndexAlphaCNG::Build(size_t n, const float *data, const Parameters &parameters) {
  std::string nn_graph_path = parameters.Get<std::string>("nn_graph_path");
  unsigned range = parameters.Get<unsigned>("R"); 
  Load_nn_graph(nn_graph_path.c_str());

  std::cout << "tau = " << tau << std::endl;
  std::cout << "alpha = " << alpha << std::endl;

  data_ = data;
  init_graph(parameters);
  SimpleNeighbor *cut_graph_ = new SimpleNeighbor[nd_ * (size_t)range];
  Link(parameters, cut_graph_);
  final_graph_.resize(nd_);

  for (size_t i = 0; i < nd_; i++) {
    SimpleNeighbor *pool = cut_graph_ + i * (size_t)range;
    unsigned pool_size = 0;
    for (unsigned j = 0; j < range; j++) {
      if (pool[j].distance == -1) break;
      pool_size = j;
    }
    pool_size++;
    final_graph_[i].resize(pool_size);
    for (unsigned j = 0; j < pool_size; j++) {
      final_graph_[i][j] = pool[j].id;
    }
  }

  {
    unsigned max = 0, min = 1e6, avg = 0;
    for (size_t i = 0; i < nd_; i++) {
      auto size = final_graph_[i].size();
      max = max < size ? size : max;
      min = min > size ? size : min;
      avg += size;
    }
    avg /= 1.0 * nd_;
    printf("Degree Statistics: Max = %d, Min = %d, Avg = %d\n", max, min, avg);
  }

  tree_grow(parameters);

  unsigned max = 0, min = 1e6, avg = 0;
  for (size_t i = 0; i < nd_; i++) {
    auto size = final_graph_[i].size();
    max = max < size ? size : max;
    min = min > size ? size : min;
    avg += size;
  }
  avg /= 1.0 * nd_;
  printf("Degree Statistics: Max = %d, Min = %d, Avg = %d\n", max, min, avg);

  has_built = true;
}




bool sortbysec(const std::pair<int,float> &a,
              const std::pair<int,float> &b)
{
    return (a.second < b.second);
}

bool sortbyNeighbor(Neighbor &a,
              Neighbor &b)
{
    return (a.distance < b.distance);
}
void IndexAlphaCNG::DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt) {
  unsigned tmp = root;
  std::stack<unsigned> s;
  s.push(root);
  if (!flag[root]) cnt++;
  flag[root] = true;
  while (!s.empty()) {
    unsigned next = nd_ + 1;
    for (unsigned i = 0; i < final_graph_[tmp].size(); i++) {
      if (flag[final_graph_[tmp][i]] == false) {
        next = final_graph_[tmp][i];
        break;
      }
    }
    if (next == (nd_ + 1)) {
      s.pop();
      if (s.empty()) break;
      tmp = s.top();
      continue;
    }
    tmp = next;
    flag[tmp] = true;
    s.push(tmp);
    cnt++;
  }
}
void IndexAlphaCNG::Search(const float *query, const float *x, size_t K,
                      const Parameters &parameters, unsigned *indices) {
  const unsigned L = parameters.Get<unsigned>("L_search");
  data_ = x;
  std::vector<Neighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);
  boost::dynamic_bitset<> flags{nd_, 0};


  unsigned tmp_l = 0;
  for (; tmp_l < L && tmp_l < final_graph_[ep_].size(); tmp_l++) {
    init_ids[tmp_l] = final_graph_[ep_][tmp_l];
    flags[init_ids[tmp_l]] = true;
  }

  while (tmp_l < L) {
    thread_local std::mt19937 rng(std::random_device{}());
    unsigned id = rng() % nd_;
    if (flags[id]) continue;
    flags[id] = true;
    init_ids[tmp_l] = id;
    tmp_l++;
  }

  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    float dist =
        distance_->compare(data_ + dimension_ * id, query, (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    // flags[id] = true;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;
        float dist =
            distance_->compare(query, data_ + dimension_ * id, (unsigned)dimension_);
        if (dist >= retset[L - 1].distance) continue;
        Neighbor nn(id, dist, true);
        int r = InsertIntoPool(retset.data(), L, nn);

        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }
}
void IndexAlphaCNG::Search_(const float *query, const float *x, size_t K,
  const Parameters &parameters, unsigned *indices,
  int gtNN, unsigned &query_iterations, unsigned &query_computations) {
    const unsigned L = parameters.Get<unsigned>("L_search");
    unsigned stopL = L;
    // bool all_visited = true;
    // bool has_unvisited_neighbors = false;
  
    data_ = x;
    std::vector<Neighbor> retset(L + 1);
    std::vector<unsigned> init_ids(L);
    boost::dynamic_bitset<> flags{nd_, 0};
  
  
  
    unsigned tmp_l = 0;
    init_ids[tmp_l] = ep_; flags[ep_] = true; tmp_l++; 
  
    for (unsigned i = 0; i < tmp_l; i++) {
      unsigned id = init_ids[i];
      float dist = distance_->compare(data_ + dimension_ * id, query, (unsigned)dimension_);
      retset[i] = Neighbor(id, dist, true);
      query_computations++;
    }
  
    for (unsigned i = tmp_l; i < init_ids.size(); i++) {
      unsigned id = init_ids[i];
      retset[i] = Neighbor(-1, 100000000.0, false);
    }
  
    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;
    int hop = 0;
    // int repeat=0;
  
    while (k < (int) stopL) {
      int nk = stopL; 
      // all_visited = true; // Reset the flag for each iteration    
      if (retset[k].flag) {
        query_iterations++;
        retset[k].flag = false;
        unsigned n = retset[k].id;
        // // Record retset[0] for this iteration
        // log_file << "Query Iteration " << query_iterations 
        //          << ", distance: " << retset[0].distance ;
      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) {
          // repeat++;
          continue;
        }
        // has_unvisited_neighbors = true; // 找到未访问邻居
        flags[id] = 1;
            
        float dist =
              distance_->compare(query, data_ + dimension_ * id, (unsigned)dimension_);
        query_computations++;
  
      if (dist >= retset[L - 1].distance) continue;
        Neighbor nn(id, dist, true);
        int r = InsertIntoPool(retset.data(), L, nn);
        if (r < nk) nk = r;
      }
        // 检查 retset 中是否还有未访问的节点
      // for (int i = 0; i < L; ++i) {
      //     if (retset[i].flag) {
      //         all_visited = false;
      //         break;
      //     }
      // }
      // if (all_visited) {
      //     // std::cout << "All nodes in retset have been visited. Exiting early." << std::endl;
      //     break;
      // }
      // log_file<< "Repeat:  " << repeat << " }\n";
      // repeat=0;
      }
      if (nk <= k)
        k = nk;
      else
        ++k;
    }
      // Close the log file
    // log_file << "===========================\n";
    // log_file.close();
      for (size_t i = 0; i < K; i++) {
      indices[i] = retset[i].id;
    }
}



void IndexAlphaCNG::findroot(boost::dynamic_bitset<> &flag, unsigned &root,
                        const Parameters &parameter) {
  unsigned id = nd_;
  for (unsigned i = 0; i < nd_; i++) {
    if (flag[i] == false) {
      id = i;
      break;
    }
  }

  if (id == nd_) return;  // No Unlinked Node

  std::vector<Neighbor> tmp, pool;
  get_neighbors(data_ + dimension_ * id, parameter, tmp, pool);
  std::sort(pool.begin(), pool.end());

  unsigned found = 0;
  for (unsigned i = 0; i < pool.size(); i++) {
    if (flag[pool[i].id]) {
      root = pool[i].id;
      found = 1;
      break;
    }
  }
  if (found == 0) {
    while (true) {
      unsigned rid = rand() % nd_;
      if (flag[rid]) {
        root = rid;
        break;
      }
    }
  }
  final_graph_[root].push_back(id);
}


void IndexAlphaCNG::tree_grow(const Parameters &parameter) {
  unsigned root = ep_;
  boost::dynamic_bitset<> flags{nd_, 0};
  unsigned unlinked_cnt = 0;
  while (unlinked_cnt < nd_) {
    DFS(flags, root, unlinked_cnt);
    if (unlinked_cnt >= nd_) break;
    findroot(flags, root, parameter);
  }
  for (size_t i = 0; i < nd_; ++i) {
    if (final_graph_[i].size() > width) {
      width = final_graph_[i].size();
    }
  }
}


float IndexAlphaCNG::eval_recall(std::vector<std::vector<unsigned> > query_res, std::vector<std::vector<int> > gts, int K){
  float mean_recall=0;
  for(unsigned i=0; i<query_res.size(); i++){
    assert(query_res[i].size() <= gts[i].size());
    
    float recall = 0;
    std::set<unsigned> cur_query_res_set(query_res[i].begin(), query_res[i].end());
    std::set<int> cur_query_gt(gts[i].begin(), gts[i].begin()+K);
    
    for (std::set<unsigned>::iterator x = cur_query_res_set.begin(); x != cur_query_res_set.end(); x++) { 
      std::set<int>::iterator iter = cur_query_gt.find(*x);
      if (iter != cur_query_gt.end()) {
        recall++;
      }
    }
    recall = recall / query_res[i].size();
    mean_recall += recall;
  }
  mean_recall = (mean_recall / query_res.size());

  return mean_recall;
}

}