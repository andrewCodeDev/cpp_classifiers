#include <iterator>
#include <map>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#define BEGIN std::cout << '\n';
#define END std::cout << "\n\n";

template<class T>
class LDA
{

private:

  int col_;
  int row_;

  int transformed_components;

  std::map<int, int> class_info;

  // container for the data buffer
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data_matrix;

  // linear map must be of complex type for eigenvector storage
  Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> linear_map;

public:

  explicit LDA (int num_samples, int num_features, int reduced_cols) :
    row_(num_samples),
    col_(num_features),
    transformed_components(reduced_cols)
    {
      data_matrix.resize(num_samples, num_features);
    }

  template<typename DATA>
  void copy_data(const DATA&);

  auto transform();

  template<typename TARGETS>
  void fit(const TARGETS&);

  template<typename TARGETS>
  void map_classes(const TARGETS&);

  template<typename MATRIX, typename TARGETS>
  void load_samples(MATRIX&, const TARGETS&, const int&);

  template<typename MATRIX>
  void populate_map(const MATRIX&, const int&, const int&);

  void print_data();
  void print_map();
  void print_info();
  
};


template<class T>
template<typename TARGET>
void LDA<T>::fit(const TARGET& targets)
{
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> scatter_b;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> scatter_w;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> class_matrix;

  scatter_b.resize(col_, col_);
  scatter_w.resize(col_, col_);
  linear_map.resize(col_, transformed_components);

  this->map_classes(targets);

  auto total_mean = data_matrix.colwise().mean();

  for(auto pair : class_info) // populate scatter matrices
  {
    class_matrix.resize(pair.second, col_);
    
    load_samples(class_matrix, targets, pair.first);

    auto class_mean = class_matrix.colwise().mean();
    auto mean_diff = class_mean - total_mean;

    scatter_w += (class_matrix.rowwise() - class_mean).transpose() * (class_matrix.rowwise() - class_mean);

    T num_rows = class_matrix.rows();

    scatter_b += num_rows * (mean_diff.transpose() * mean_diff);

  }

  auto eig_raw = scatter_w.inverse() * scatter_b;

  Eigen::EigenSolver<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eig_ref(eig_raw);

  populate_map(eig_ref.eigenvectors(), col_, transformed_components);
  
}


template<class T>
template<typename TARGETS>
void LDA<T>::map_classes(const TARGETS& targets)
{
  for(auto x : targets)
  {
    if (class_info.contains(x))
      class_info[x] += 1;
    else
      class_info[x] = 1;
  }
}


template<class T>
template<typename DATA>
void LDA<T>::copy_data(const DATA& arr)
{
    auto aptr = arr.begin();
    auto mptr = &data_matrix(0);

  do{

    *mptr = *aptr;

    std::advance(mptr, 1);
    std::advance(aptr, 1);
    
  }while(aptr != arr.end());
}


template<class T>
template<typename MATRIX, typename TARGETS>
void LDA<T>::load_samples(MATRIX& mat, const TARGETS& targets, const int& id)
{
  int j, x = 0;

  for(int i = 0; i < row_; ++i)
  {
    if (targets[i] == id)
    {
      for(j = 0; j < col_; ++j)
      {
        mat(x, j) = data_matrix(i, j);
      }
      ++x;
    }
  }
}


template<class T>
template<typename MATRIX>
void LDA<T>::populate_map(const MATRIX& mat, const int& row, const int& col)
{
  for(int i = 0; i < row; ++i)
    for(int j = 0; j < col; ++j)
      this->linear_map(i, j) = mat(i, j);
}


template <class T>
auto LDA<T>::transform()
{
  return data_matrix * linear_map;
}


template<class T>
void LDA<T>::print_info()
{
  for(auto x : class_info)
  {
    std::cout << "Key " << x.first << ": " << x.second << '\n';
  }
  std::cout << '\n';
}


template <class T>
void LDA<T>::print_data()
{
  std::cout <<  data_matrix;
  std::cout << "\n\n";
}


template <class T>
void LDA<T>::print_map()
{
  std::cout << linear_map;
  std::cout << "\n\n";
}
