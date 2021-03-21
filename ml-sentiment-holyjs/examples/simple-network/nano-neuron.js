// @ts-check

function random(start, end) {
  return Math.random() * (end - start) + start;
}

function celsiusToFahrenheit(celsius) {
  return 1.8 * celsius + 32;
}

function generateDataSets() {
  const xTrain = [];
  const yTrain = [];
  for (let x = 0; x < 100; x += 1) {
    const y = celsiusToFahrenheit(x);
    xTrain.push(x);
    yTrain.push(y);
  }

  const xTest = [];
  const yTest = [];
  for (let x = 0.5; x < 100; x += 1) {
    const y = celsiusToFahrenheit(x);
    xTest.push(x);
    yTest.push(y);
  }

  return [xTrain, yTrain, xTest, yTest];
}

function predictionCost(y, prediction) {
  return (y - prediction) ** 2 / 2;
}

class SimpleNeuralNetwork {
  constructor(a = random(-100, 100), b = random(-100, 100)) {
    this.a = a;
    this.b = b;
  }

  train({ epochs, alpha, xTrain, yTrain }) {
    const costHistory = [];

    for (let epoch = 0; epoch < epochs; epoch += 1) {
      const [predictions, cost] = this.forwardPropagation(xTrain, yTrain);
      costHistory.push(cost);

      const [dW, dB] = this.backwardPropagation(predictions, xTrain, yTrain);
      this.a += alpha * dW;
      this.b += alpha * dB;
    }

    return costHistory;
  }

  predict(x) {
    const { a, b } = this;
    return a * x + b;
  }

  run() {
    // Generate training and test data-sets.
    const [xTrain, yTrain, xTest, yTest] = generateDataSets();

    // Let's train the model with small (0.0005) steps during the 70000 epochs.
    // You can play with these parameters, they are being defined empirically.
    const epochs = 70000;
    const alpha = 0.0005;
    const trainingCostHistory = this.train({
      epochs,
      alpha,
      xTrain,
      yTrain,
    });

    // Let's check how the cost function was changing during the training.
    // We're expecting that the cost after the training should be much lower than before.
    // This would mean that NanoNeuron got smarter. The opposite is also possible.
    console.log('Cost before the training:', trainingCostHistory[0]); // i.e. -> 4694.3335043
    console.log('Cost after the training:', trainingCostHistory[epochs - 1]); // i.e. -> 0.0000024

    // Let's take a look at NanoNeuron parameters to see what it has learned.
    // We expect that NanoNeuron parameters 'w' and 'b' to be similar to ones we have in
    // celsiusToFahrenheit() function (w = 1.8 and b = 32) since our NanoNeuron tried to imitate it.
    console.log('NanoNeuron parameters:', {
      a: this.a,
      b: this.b,
    }); // i.e. -> {w: 1.8, b: 31.99}

    // Evaluate our model accuracy for test data-set to see how well our NanoNeuron deals with new unknown data predictions.
    // The cost of predictions on test sets is expected to be be close to the training cost.
    // This would mean that NanoNeuron performs well on known and unknown data.
    const [testPredictions, testCost] = this.forwardPropagation(xTest, yTest);
    console.log('Cost on new testing data:', testCost); // i.e. -> 0.0000023

    // Now, since we see that our NanoNeuron "kid" has performed well in the "school" during the training
    // and that he can convert Celsius to Fahrenheit temperatures correctly even for the data it hasn't seen
    // we can call it "smart" and ask him some questions. This was the ultimate goal of whole training process.
    const tempInCelsius = 70;
    const customPrediction = this.predict(tempInCelsius);
    console.log(
      `NanoNeuron "thinks" that ${tempInCelsius}°C in Fahrenheit is:`,
      customPrediction,
    ); // -> 158.0002
    console.log('Correct answer is:', celsiusToFahrenheit(tempInCelsius)); // -> 158
  }

  // Forward propagation.
  // This function takes all examples from training sets xTrain and yTrain and calculates
  // model predictions for each example from xTrain.
  // Along the way it also calculates the prediction cost (average error our NanoNeuron made while predicting).
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

  // Backward propagation.
  // This is the place where machine learning looks like a magic.
  // The key concept here is derivative which shows what step to take to get closer
  // to the function minimum. Remember, finding the minimum of a cost function is the
  // ultimate goal of training process. The cost function looks like this:
  // (y - prediction) ^ 2 * 1/2, where prediction = x * w + b.
  backwardPropagation(predictions, xTrain, yTrain) {
    const m = xTrain.length;
    // At the beginning we don't know in which way our parameters 'w' and 'b' need to be changed.
    // Therefore we're setting up the changing steps for each parameters to 0.
    let dW = 0;
    let dB = 0;
    for (let i = 0; i < m; i += 1) {
      // This is derivative of the cost function by 'w' param.
      // It will show in which direction (positive/negative sign of 'dW') and
      // how fast (the absolute value of 'dW') the 'w' param needs to be changed.
      dW += (yTrain[i] - predictions[i]) * xTrain[i];
      // This is derivative of the cost function by 'b' param.
      // It will show in which direction (positive/negative sign of 'dB') and
      // how fast (the absolute value of 'dB') the 'b' param needs to be changed.
      dB += yTrain[i] - predictions[i];
    }
    // We're interested in average deltas for each params.
    dW /= m;
    dB /= m;
    return [dW, dB];
  }
}

module.exports = SimpleNeuralNetwork;

// @ts-ignore
const isMainModule = typeof require !== 'undefined' && require.main === module;

if (isMainModule) {
  const w = Math.random(); // i.e. -> 0.9492
  const b = Math.random(); // i.e. -> 0.4570
  const neuralNetwork = new SimpleNeuralNetwork(w, b);
  neuralNetwork.run();
}
