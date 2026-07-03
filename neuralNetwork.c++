class NeuralNetwork {
  private:
    std::vector<Layer> layers;

  public: 
    NeuralNetwork(const std::vector<Layer> layers) : layers(layers) {}

    VectorXd forward(VectorXd input) {
      for (Layer& layer : layers) {
        layer.forward(input);
        input = layer.get_last_values();
      }
      return input;
    }


    VectorXd get_output() {
      return layers.back().get_last_values();
    }

    void backward(VectorXd& input, VectorXd& target_output, double learning_hate){
      VectorXd output_error = error_deritive(get_output(), target_output);
      VectorXd output_derivative = layers.back().dericative_function();
      VactorXd Output_delta = output_error.cwiseProduct(output_derivative);
      layers.back().set_delta(Output_delta);

      for int(i = layers.size() - 2; i >= 0; i--){
        MatrixXd weights = layers[i + 1].get_weights();
        VectorXd delta = layers[i + 1].get_delta();

        VectorXd hidden_error =  weights * delta;
        VectorXd hidden_derivative = layers[i+1].dericative_function();
        VectorXd hidden_delta = hidden_error.cwiseProduct(hidden_derivative);
        layer[i].set_delta(hidden_delta)
      }

      for (int i = 0; i < layers.size(); i++){
        VextorXd input_;
        if(i == 0){
          input_ = input
        } else {
          input = layer[i-1].get_neurons_values_activated();
        }

        layers[i].update_weights(learning_hate, input_);
      }
}