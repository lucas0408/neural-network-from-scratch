#include <iostream>
using namespace std;
using namespace Eigen;

class layer {
  MatrixXd weights;
  VectorXd biases;
  VectorXd neurons_values;
  VectorXd neurons_values_activated;
  std::function<VectorXd(const VectorXd&)> activation_function;
  std::function<VectorXd(const VectorXd&)> activation_function_derivative;

  VectorXd delta;


  public:
    layer(int input_size, int output_size, std::function<VectorXd(const VectorXd&)> activation_function,
         std::function<VectorXd(const VectorXd&)> activation_function_derivative) {
      weights = MatrixXd::Random(output_size, input_size);
      biases = VectorXd::Random(output_size);
      this->activation_function = activation_function;
      this->activation_function_derivative = activation_function_derivative;
    }

    VectorXd get_weights() {
      return weights;
    }

    Vector get_biases() {
      return biases;
    }

    Vector get_neurons_values(){
      return neurons_values;
    }

    VectorXd get_neurons_values_activated(){
      return neurons_values_activated;
    }

    void set_delta(VectorXd delta){
      this->delta = delta;
    }

    void forward(const VectorXd& input) {
      neurons_values = (weights * input) + biases;
      neurons_values_activated = activation_function(neurons_values);
    }

    void update_weights(double learning_hate, vectorXd &i){
      weights = weights - learning_hate * (i * delta.transpose());
      biases = biases - learning_hate * delta;
    }

    VectorXd dericative_function(){
      return activation_function_derivative(neurons_values);
    }

    VectorXd get_last_values() {
      return neurons_values_activated;
    }
}