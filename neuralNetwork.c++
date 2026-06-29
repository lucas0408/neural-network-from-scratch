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

}