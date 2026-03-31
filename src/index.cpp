
#include <efanna2e/index.h>
namespace efanna2e {
Index::Index(const size_t dimension, const size_t n, Metric metric = INNER_PRODUCT)
  : dimension_ (dimension), nd_(n), has_built(false) {
    switch (metric) {
      case L2:distance_ = new DistanceL2();
        break;
      case INNER_PRODUCT: distance_ = new DistanceInnerProduct(); 
        break;
      default:distance_ = new DistanceL2();
        break;
    }
}

Index::~Index() {}
}
