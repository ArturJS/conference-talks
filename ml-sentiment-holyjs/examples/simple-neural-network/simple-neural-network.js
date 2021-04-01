// @ts-check

const {
  celsiusToFahrenheit,
  generateDataSets,
  predictionCost,
  random,
} = require('./utils');

class SimpleNeuralNetwork {
  constructor(a, b) {
    this.a = a;
    this.b = b;
    this.logger = {
      info: console.info,
    };
  }

  train({ epochs, alpha, xTrain, yTrain }) {
    const costHistory = [];

    for (let epoch = 0; epoch < epochs; epoch += 1) {
      const [predictions, cost] = this.forwardPropagation(xTrain, yTrain);
      costHistory.push(cost);

      const [dA, dB] = this.backwardPropagation(predictions, xTrain, yTrain);
      this.a += alpha * dA;
      this.b += alpha * dB;
    }

    return costHistory;
  }

  predict(x) {
    const { a, b } = this;
    return a * x + b;
  }

  run() {
    const [xTrain, yTrain, xTest, yTest] = generateDataSets();
    const epochs = 70000;
    const alpha = 0.0005;
    const trainingCostHistory = this.train({
      epochs,
      alpha,
      xTrain,
      yTrain,
    });

    this.logger.info('Cost before the training:', trainingCostHistory[0]); // i.e. -> 4694.3335043
    this.logger.info(
      'Cost after the training:',
      trainingCostHistory[epochs - 1],
    ); // i.e. -> 0.0000024
    this.logger.info('NanoNeuron parameters:', {
      a: this.a,
      b: this.b,
    }); // i.e. -> {a: 1.8, b: 31.99}

    const [testPredictions, testCost] = this.forwardPropagation(xTest, yTest);
    this.logger.info('Cost on new testing data:', testCost); // i.e. -> 0.0000023

    const tempInCelsius = 70;
    const customPrediction = this.predict(tempInCelsius);
    this.logger.info(
      `NanoNeuron "thinks" that ${tempInCelsius}°C in Fahrenheit is:`,
      customPrediction,
    ); // -> 158.0002
    this.logger.info('Correct answer is:', celsiusToFahrenheit(tempInCelsius)); // -> 158
  }

  forwardPropagation(xTrain, yTrain) {
    const m = xTrain.length;
    const predictions = [];
    let cost = 0;

    for (let i = 0; i < m; i += 1) {
      const prediction = this.predict(xTrain[i]);
      cost += predictionCost(yTrain[i], prediction);
      predictions.push(prediction);
    }
    // We are interested in average cost.
    cost /= m;
    return [predictions, cost];
  }

  backwardPropagation(predictions, xTrain, yTrain) {
    const m = xTrain.length;
    let dA = 0;
    let dB = 0;
    for (let i = 0; i < m; i += 1) {
      dA += (yTrain[i] - predictions[i]) * xTrain[i];
      dB += yTrain[i] - predictions[i];
    }
    // We're interested in average deltas for each params.
    dA /= m;
    dB /= m;
    return [dA, dB];
  }
}

module.exports = SimpleNeuralNetwork;

// @ts-ignore
const isMainModule = typeof require !== 'undefined' && require.main === module;

if (isMainModule) {
  const a = Math.random(); // i.e. -> 0.9492
  const b = Math.random(); // i.e. -> 0.4570
  const neuralNetwork = new SimpleNeuralNetwork(a, b);
  neuralNetwork.run();
}

// see also https://github.com/trekhleb/nano-neuron/blob/master/NanoNeuron.js
