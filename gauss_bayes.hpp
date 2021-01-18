#pragma once

#include <unordered_map>
#include <utility>
#include <concepts>
#include <numbers>
#include <cmath>

#include <assert.h>

#include <Eigen/Dense>


 // Free helper functions ~~~~~~~~~~~~~~~~~~~~

template <typename ID, std::floating_point T>
class GaussianBayes
{
private:

  std::unordered_map<ID, std::pair<int, T>> class_map;

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data_matrix;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mean_matrix;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> var_matrix;

public:

  GaussianBayes() = default;

  int get_class_num(const ID&) const;
  T get_class_prior(const ID&) const;

  void add_class_num(const ID&, const int&);
  void add_class_prior(const ID&, const T&);

  template <typename TARGETS>
  void load_class_map(const TARGETS&); // use Y

  template <typename TRAINING, typename TARGETS>
  void fit(const TRAINING&, const TARGETS&, const int&);

  template <typename TRAIN, typename TARGETS>
  void load_array(const TRAIN&, const TARGETS&, const int& id);

  template <std::integral CLASSID, typename VECTOR>
  T gaussian_posterior(const CLASSID&, const VECTOR&);

  template <typename VECTOR>
  ID predict(const VECTOR&);

  ID retrive_class(const int&);

}; // END CLASS 


// helper functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


template <typename ID, std::floating_point T>
int GaussianBayes<ID, T>::get_class_num(const ID& target) const
{
  return class_map[target].first;
}

template <typename ID, std::floating_point T>
T GaussianBayes<ID, T>::get_class_prior(const ID& target) const
{
  return class_map[target].second;
}

template <typename ID, std::floating_point T>
void GaussianBayes<ID, T>::add_class_num(const ID& target, const int& x)
{
  class_map[target].first += x;
}

template <typename ID, std::floating_point T>
void GaussianBayes<ID, T>::add_class_prior(const ID& target, const T& x)
{
  class_map[target].second += x;
}


// calculation functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


template <typename ID, std::floating_point T>
template <typename TRAINING, typename TARGETS>
void GaussianBayes<ID, T>::fit(const TRAINING& train, const TARGETS& target, const int& n_features)
{
  load_class_map(target);

  mean_matrix.resize(class_map.size(), n_features);
  var_matrix.resize(class_map.size(), n_features);

  for (auto id : class_map)
  {
    data_matrix.resize(get_class_num(id.first, n_features));
    load_data_matrix(train, target, id.first);

  // load the matrices with the mean and variance outputs
    mean_matrix.row(id.first) = data_matrix.colwise().mean();
    var_matrix.row(id.first)  = data_matrix.array().pow(2).colwise().sum();
  }
}


template<typename ID, std::floating_point T>
template <typename VECTOR>
ID GaussianBayes<ID, T>::predict(const VECTOR& x)
{
  T prediction{0.0}, likelihood{0.0};
  int idx;

  for (int id{0}; id < class_map.size(); ++id)
  {
    likelihood = gaussian_posterior(id, x) + log(get_class_prior(id));
    if (prediction < likelihood)
    {
      prediction = likelihood;
      idx = id;
    }
  }
  return this->retrieve_class(idx);
}


template<typename ID, std::floating_point T>
template <std::integral CLASSID, typename VECTOR>
T GaussianBayes<ID, T>::gaussian_posterior(const CLASSID& idx, const VECTOR& x)
{
  auto numerator   =  Eigen::exp( - (x.array() - mean_matrix.row(idx).array()).pow(2));
  auto denominator =  (2 * std::numbers::pi_v<T> * var_matrix.row(idx).array()).sqrt();
  return Eigen::log( numerator.array() / denominator.array() ).rowwise().sum();
}


// memory functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


template <typename ID, std::floating_point T>
ID GaussianBayes<ID, T>::retrive_class(const int& idx)
{
  return std::next(class_map.begin(), (class_map.size() - 1) - idx)->first;
}

template <typename ID, std::floating_point T>
template <typename TARGETS>
void GaussianBayes<ID, T>::load_class_map(const TARGETS& targets)
{
  T n{0.0};

  for (auto target : targets)
  {
    ++n;
    add_class_num(target, 1);
    add_class_prior(target, static_cast<T>(1.0));
    
  }

  for (auto& p : class_map)
  {
    p.second.second /= n;
  }
}

template <typename ID, std::floating_point T>
template<typename TRAIN, typename TARGETS>
void GaussianBayes<ID, T>::load_array(const TRAIN& train, const TARGETS& targets, const int& id)
{
  int row{0}, col{0}, cols = data_matrix.cols();

  auto data_ptr = &data_matrix(0);

  for(auto target : targets)
  {
    if (target == id)
    {
      for(col = 0; cols < cols; ++col)
      {
        *data_ptr = train[cols * row + col];
        std::advance(data_ptr, 1);
      }
    }
    ++row;
  }
}