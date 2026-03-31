# Fast-Convergent-Proximity-Graphs-for-Approximate-Nearest-Neighbor-Search

This repository contains the implementation of **$\alpha$-CNG**, a proximity graph-based indexing algorithm for efficient Approximate Nearest Neighbor (ANN) search. The algorithm is designed to support fast convergence during beam search by adaptively pruning connections, offering a balance between high recall and low computation.
## Building Instruction
### Prerequisites
The implementation requires:
+ GCC 4.9+ with OpenMP
+ CMake 2.8+
+ Boost 1.55+
+ [TCMalloc](http://goog-perftools.sourceforge.net/doc/tcmalloc.html)
### Compile On Ubuntu/Debian
1. Install Dependencies:
```shell
$ sudo apt-get install g++ cmake libboost-dev libgoogle-perftools-dev
```
2. Compile Alpha-CNG:
```shell
$ cd Fast-Convergent-Proximity-Graphs-for-Approximate-Nearest-Neighbor-Search/
$ mkdir build/ && cd build/
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make -j
```
## Usage
$\alpha$-CNG follows a similar construction pipeline to NSG. It first builds an approximate k-NN graph and then prunes it into a sparse proximity graph with alpha-reducible properties.
The usage is divided into two parts:
- **Construction**: Build and convert a kNN graph into an Alpha-CNG index
- **Search**: Perform beam search on the constructed Alpha-CNG graph
- ---
### Construction
#### **Step 1: Build Approximate k-NN Graph**
You can use any method to construct the initial approximate k-NN graph.  In our experiments, we use [EFANNA](https://github.com/ZJULearning/efanna_graph) to generate the kNN graph. The related parameters are listed in the **Parameter** section below.
#### **Step 2: Convert kNN Graph to $\alpha$-CNG**
This step converts the dense k-NN graph into a sparse $\alpha$-CNG with better search efficiency and convergence behavior.
```bash
cd build/tests/
./test_alphacng_index DATA_PATH KNNG_PATH L R C ALPHA_START ALPHA_STEP ALPHA_MAX TAU T SAVE_GRAPH_PATH
```
**Arguments:**
- `DATA_PATH` is the path to the base dataset vectors, typically in fvecs format.
- `KNNG_PATH` is the path to the pre-built approximate kNN graph from _Step 1_.
- `L` controls the beam width (queue size) during Alpha-CNG construction; larger values generally improve quality but increase computation.
- `R` controls the maximum out-degree (number of neighbors per node) in the final graph; it affects both index size and connectivity.
- `C` sets the candidate pool size used during pruning; a larger value may retain better neighbors but increases overhead.
- `ALPHA_START` is the initial α value used for adaptive pruning.
- `ALPHA_STEP` is the step size for increasing α when the node's degree is below the threshold.
- `ALPHA_MAX` is the maximum α value allowed during adaptive pruning.
- `TAU` is assumed upper bound on the distance between a query point and its nearest neighbor; this assumption guides pruning but does not need to strictly hold in practice.
- `T` is the node degree threshold; if a node’s degree after pruning is less than `T`, $\alpha$ will be increased to preserve connectivity.
- `SAVE_GRAPH_PATH` is the output path where the constructed Alpha-CNG index will be stored.

### Search 
After building the Alpha-CNG graph, you can perform ANN search using beam search:  
```bash
cd build/tests/
./test_alphacng_search DATA_PATH QUERY_PATH GT_PATH ALPHA_CNG_PATH SEARCH_L SEARCH_K
```
- `DATA_PATH` is the path to the base dataset vectors.
- `QUERY_PATH` is the path to the query vectors, typically in fvecs format
- `GT_PATH` is the path to the ground truth file used for evaluating recall or other metrics.
- `ALPHA_CNG_PATH` is the path to the prebuilt Alpha-CNG graph.
- `SEARCH_L` is the beam width (queue size) used during beam search; larger values typically improve recall but increase computation.
- `SEARCH_K` is the number of nearest neighbors to retrieve for each query.

After execution, the program will print **key search performance** metrics to standard output, including:
- Total and average **distance computations**
- Total and average number of **hops** during beam search
- Total **elapsed time**
- Final **Recall@k** (based on the provided ground truth)

##  Parameters Used in Our Paper
This section summarizes the key parameters we used in our experiments across multiple datasets.  
The construction process involves two stages:
### Approximate K-NN Graph Construction
We used [EFANNA](https://github.com/ZJULearning/efanna_graph) to construct 200-NN graphs for all datasets. Specifically, the parameters used to construct approximate KNN graph are as follows: `K = 200, L = 200, iter = 10, S = 10, R = 100`.For the detailed explanation of these parameters, please refer to the original [EFANNA documentation](https://github.com/ZJULearning/efanna_graph).

### Alpha-CNG Construction Parameters
In all experiments, we use the following adaptive pruning configuration: `alpha_start = 0.9 （alpha_start=1 for bigann100M）`, `alpha_step = 0.05`, `alpha_max = 1.6`, `T=R/2`
The following table summarizes the construction parameters (`tau`, `L`, `R`, and `C`) used for each dataset:
| Dataset    | Tau | L   | R   | C   |
|------------|-----|-----|-----|-----|
| SIFT       | 10   | 40  | 50  | 500 |
| CRAWL      | 0.15   | 100 | 100 | 500 |
| WIKI       | 0.1   | 60  | 70  | 500 |
| MSONG      | 3.0   | 100 | 100 | 500 |
| LAIONI2I   | 0.008   | 60  | 70  | 500 |
| GIST       | 0.02   | 60  | 70  | 500 |
| DEEP100M       | 0.04   | 100  | 100  | 500 |
| BIGANN100M       | 10   | 80  | 100  | 500 |

## Example: Run Alpha-CNG on GIST Dataset
This section walks through a full example of running $\alpha$-CNG on the GIST dataset. 
```bash
# === Step 1: Download and Extract GIST Dataset ===
wget ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz           # download dataset
tar -xzvf gist.tar.gz                                              # extract .fvecs and .ivecs files

# Files extracted:
# - gist_base.fvecs       # base vectors (1M x 960D)
# - gist_query.fvecs      # query vectors (1000 x 960D)
# - gist_groundtruth.ivecs# ground-truth (K=100)

# === Step 2: Build Approximate kNN Graph Using EFANNA ===
# Follow the build instructions from: https://github.com/ZJULearning/efanna_graph
efanna_graph/tests/test_nndescent gist.fvecs gist_200nn.graph 200 200 10 10 100  # K=200, L=200, iter=10, S=10, R=100

# === Step 3: Construct Alpha-CNG Index ===
# Compile the Alpha-CNG project
cd Fast-Convergent-Proximity-Graphs-for-Approximate-Nearest-Neighbor-Search/
mkdir build/ && cd build/
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j

# Note: Replace "/path/to/..." with your actual file paths
cd tests
./test_alphacng_index  /path/to/gist_base.fvecs /path/to/gist_200nn.graph 60 70 500 0.9 0.05 1.6 0.06 35  gist_alphaCNG.graph 

# === Step 4: Run Beam Search on Alpha-CNG ===
# Note: Replace "/path/to/..." with your actual file paths
./test_alphacng_search /path/to/gist_base.fvecs /path/to/gist_query.fvecs /path/to/gist_groundtruth.ivecs gist_alphaCNG.grarh 300 100                      
```

Here is an example output of Alpha-CNG Search with `L=300 K=100`
```bash
data dimension: 960              # Dimensionality of base vectors
data dimension: 960              # Dimensionality of query vectors
data dimension: 100              # Ground truth size (top-K)
Elapsed time: 5.24994 seconds.   # Total search time for all queries
Total computations: 11946352     # Total number of distance computations
Total iterations: 303237         # Total number of hops during beam search
Average computations per query: 11946.4   # Average number of distance 
Average iterations per query: 303.237     # Average number of hops 
recall 0.987198                 # Recall@100 
```
