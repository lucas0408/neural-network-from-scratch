# Neural Network em C++ (Eigen)

Implementação de uma rede neural feedforward (MLP) do zero em C++, utilizando a biblioteca [Eigen](https://eigen.tuxfamily.org/) para álgebra linear. O projeto implementa manualmente forward propagation, backpropagation via regra da cadeia e atualização de pesos por gradiente descendente — sem uso de frameworks de machine learning prontos.

## Arquitetura

A rede é modelada como uma composição de camadas (`Layer`), cada uma encapsulando seus próprios parâmetros e responsável apenas pelas operações locais a ela. A `NeuralNetwork` não conhece os detalhes internos de cada camada; ela apenas orquestra a sequência de chamadas (forward camada a camada, e backward em ordem reversa), o que mantém o acoplamento baixo e facilita trocar o número/tipo de camadas sem alterar a lógica de propagação.

### `Layer`

Cada camada guarda:

- **`weights`** (`MatrixXd`, `output_size x input_size`): matriz de pesos, inicializada aleatoriamente. O dimensionamento `output_size x input_size` permite calcular a pré-ativação diretamente como `weights * input`, sem transposições.
- **`biases`** (`VectorXd`): viés somado após a multiplicação matricial.
- **`neurons_values`**: valor de pré-ativação (`z = W·x + b`), guardado separadamente do valor ativado porque é necessário no cálculo da derivada da ativação durante o backward.
- **`neurons_values_activated`**: saída da camada após aplicar a função de ativação (`a = f(z)`), que é o valor efetivamente repassado para a próxima camada.
- **`activation_function` / `activation_function_derivative`**: funções injetadas via `std::function`, permitindo usar a mesma classe `Layer` com qualquer ativação (ReLU, sigmoid, tanh, etc.) sem precisar de subclasses.
- **`delta`**: erro local da camada (gradiente do erro em relação à pré-ativação `z`), calculado no backward e usado tanto para propagar o erro para a camada anterior quanto para atualizar os próprios pesos.

O `forward` de uma camada implementa diretamente a equação `a = f(W·x + b)`, guardando `z` e `a` para uso posterior no backward.

### `NeuralNetwork`

- **`forward`**: itera sobre as camadas, repassando a saída ativada de uma camada como entrada da próxima. O resultado final é a saída da última camada.
- **`backward`**: implementa backpropagation clássico:
  1. Calcula o erro na camada de saída comparando a predição com o `target_output`.
  2. Multiplica esse erro (element-wise, via `cwiseProduct`) pela derivada da função de ativação na última camada, obtendo o delta de saída — aplicação direta da regra da cadeia para a última camada.
  3. Para as camadas ocultas, propaga o erro de trás para frente: o erro de uma camada `i` é obtido multiplicando a matriz de pesos da camada seguinte (`i+1`) pelo delta dela (propagação do gradiente "para trás" pela rede), e então aplicando a derivada local da ativação da camada `i`.

Essa separação entre "erro" (gradiente vindo da camada seguinte) e "delta" (erro já multiplicado pela derivada local) segue a formulação padrão de backprop, deixando explícito em qual ponto a regra da cadeia está sendo aplicada.

## Decisões de design

- **Injeção de função de ativação**: em vez de hardcodar sigmoid/ReLU, a camada recebe a função e sua derivada no construtor. Isso evita duplicação de código e permite misturar ativações diferentes por camada.
- **Eigen para álgebra linear**: operações vetorizadas (`cwiseProduct`, multiplicação matricial) em vez de loops manuais, o que é mais eficiente e expressa melhor a matemática por trás do algoritmo.
- **Camadas com estado próprio**: cada `Layer` guarda seu próprio `delta`, `neurons_values` e `neurons_values_activated`, evitando que a `NeuralNetwork` precise manter estruturas paralelas para rastrear esses valores por camada.

## Funcionalidades

- [x] Forward propagation através de múltiplas camadas
- [x] Inicialização aleatória de pesos e biases
- [x] Suporte a funções de ativação customizáveis (e suas derivadas)
- [x] Cálculo de delta da camada de saída e propagação reversa do erro (backprop)
- [x] Atualização de pesos via gradiente descendente (método `update_weights` já existe na `Layer`, falta integrar ao loop de `backward`)
- [x] Função de erro/custo configurável (atualmente `error_function` é um placeholder)
- [ ] Treinamento em batch/epochs
- [ ] Testes e exemplos de uso (ex: XOR, MNIST)

## Dependências

- C++17 ou superior
- [Eigen](https://eigen.tuxfamily.org/) (biblioteca de álgebra linear, *header-only*)

## Estrutura do projeto

```
.
├── include/
│   ├── layer.hpp
│   └── neural_network.hpp
├── src/
└── README.md
```

*(ajuste essa estrutura conforme a organização final dos arquivos)*

## Como compilar

```bash
g++ -std=c++17 -I /caminho/para/eigen main.cpp -o neural_network
```

## Roadmap

1. Corrigir e finalizar o `backward()` (cálculo de erro de saída, propagação dos deltas, atualização de pesos)
2. Implementar função de erro (ex: MSE)
3. Implementar `update_weights` chamado para todas as camadas após o backward
4. Adicionar exemplos de treinamento (ex: porta lógica XOR)
5. Adicionar testes unitários

## Licença

Defina aqui a licença do projeto (ex: MIT).