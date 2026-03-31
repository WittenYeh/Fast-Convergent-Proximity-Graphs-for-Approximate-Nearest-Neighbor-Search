#include <efanna2e/index_alphacng.h>
#include <util.h>
#include <set>
#include <map>
#include <bits/stdc++.h> 
#include <math.h>
#include <sys/stat.h> // For mkdir
#include <sys/types.h> // For mode_t
#include <cstring> // For strerror

void load_data(const char* filename, float*& data, unsigned& num,
               unsigned& dim) {  
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
  std::cout << "data dimension: " << dim << std::endl;
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new float[num * dim * sizeof(float)];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
}


std::vector<std::vector<int> > load_ground_truth(const char* filename) {
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error (in load_ground_truth)" << std::endl;
    exit(-1);
  }

  unsigned dim, num;

  in.read((char*)&dim, 4);
  std::cout << "data dimension: " << dim << std::endl;
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);

  int* data = new int[num * dim * sizeof(int)];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();

  std::vector<std::vector<int> > res;
  for (int i = 0; i < num; i++) {
    std::vector<int> a;
    for (int j = i*dim; j < (i+1)*dim; j++) {
      a.push_back(data[j]);
    }
    res.push_back(a);
  }

  return res;
}


int main(int argc, char** argv) {
  if (argc != 7) {
    std::cout << argv[0]
              << " data_file query_file gt_file AlphaCNG_path search_L search_K"
              << std::endl;
    exit(-1);
  }

  std::string dataFileName = argv[1];
  std::string queryFileName = argv[2];
  std::string gtFileName = argv[3];
  std::string graph_file = argv[4];
  unsigned L = (unsigned)atoi(argv[5]);
  unsigned K = (unsigned)atoi(argv[6]);

  float* data_load = NULL;
  unsigned points_num, dim;
  load_data(dataFileName.c_str(), data_load, points_num, dim);

  float* query_load = NULL;
  unsigned query_num, query_dim;
  load_data(queryFileName.c_str(), query_load, query_num, query_dim);
  assert(dim == query_dim);

  if (L < K) {
    std::cout << "search_L cannot be smaller than search_K!" << std::endl;
    exit(-1);
  }

  efanna2e::IndexAlphaCNG index(dim, points_num, efanna2e::L2, nullptr);
  index.Load(argv[4]);
  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);

  std::vector<std::vector<int> > gts = load_ground_truth(gtFileName.c_str());

  auto s = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<unsigned> > res;
  std::vector<float> recalls; 
  std::vector<unsigned> iterations;
  std::vector<unsigned> computations;

  int qcnt = 0;
  int total_computations=0;
  int total_iterations=0;
  for (unsigned i = 0; i < query_num; i++) {

    std::vector<unsigned> tmp(K);
    unsigned query_iterations = 0;      // Track iterations for this query
    unsigned query_computations = 0;    // Track distance computations
    index.Search_(query_load + i * dim, data_load, K, paras, tmp.data(), gts[i][0],query_iterations, query_computations);

    res.push_back(tmp);
    float query_recall = index.eval_recall({tmp}, {gts[i]}, K);
    total_computations=total_computations+query_computations;
    total_iterations=total_iterations+query_iterations;
    recalls.push_back(query_recall);
    iterations.push_back(query_iterations);
    computations.push_back(query_computations);
    qcnt++;
  }
  
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;

  std::cout << "Elapsed time: " << diff.count() << " seconds." << std::endl;
  std::cout << "Total computations: " << total_computations << std::endl;
  std::cout << "Total iterations: " << total_iterations << std::endl;
  std::cout << "Average computations per query: " << (float)total_computations / qcnt << std::endl;
  std::cout << "Average iterations per query: " << (float)total_iterations / qcnt << std::endl;

  float recall = index.eval_recall(res, gts, K);
  std::cout << "recall " << recall << std::endl;
  return 0;
}