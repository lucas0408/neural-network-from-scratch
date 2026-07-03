namespace functions {
  VectorXd sigmoid(const VectorXd &v){
    return 1.0 / (1.0 + (-v.array()).exp());
  }

  VectorXd sigmoid_derivative(const VectorXd &v){
    VectorXd sigmoid = sigmoid(v);
    return sigmoid.array()*(1-sigmoid.array());
  }

  VectorXd error_function(const VectorXd &z, const VectorXd &t){
    return 0.5 * (z-t).squaredNorm();
  }

  VectorXd error_deritive(const VectorXd &z, const VectorXd &t){
    return z - t;
  }
}