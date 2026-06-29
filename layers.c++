#include <iostream>
using namespace std;
using namespace Eigen;

class layer {
  MatrixXd weights;
  VectorXd biases;
  VectorXd neurons_values;
  VectorXd neurons_values_activated;
  std::function<VectorXd(const VectorXd&)> activation_function;

  VectorXd delta;


  public:
    layer(int input_size, int output_size, std::function<VectorXd(const VectorXd&)> activation_function) {
      weights = MatrixXd::Random(output_size, input_size);
      biases = VectorXd::Random(output_size);
      this->activation_function = activation_function;
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

    VectorXd get_last_values() {
      return neurons_values_activated;
    }
}