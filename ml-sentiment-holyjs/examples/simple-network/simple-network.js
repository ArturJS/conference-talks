// ToDo implement https://github.com/trekhleb/nano-neuron/blob/master/NanoNeuron.js

function сelsiusToFahrenheit(celsius) {
  const a = 9 / 5;
  const b = 32;
  return a * celsius + b;
}

function random(start, end) {
  return Math.random() * (end - start) + start;
}

function generateDataSets() {
  // Generate TRAINING examples.
  // We will use this data to train our NanoNeuron.
  // Before our NanoNeuron will grow and will be able to make decisions by its own
  // we need to teach it what is right and what is wrong using training examples.
  // xTrain -> [0, 1, 2, ...],
  // yTrain -> [32, 33.8, 35.6, ...]
  const xTrain = [];
  const yTrain = [];
  for (let x = 0; x < 100; x += 1) {
    const y = celsiusToFahrenheit(x);
    xTrain.push(x);
    yTrain.push(y);
  }

  // Generate TEST examples.
  // This data will be used to evaluate how well our NanoNeuron performs on the data
  // that it didn't see during the training. This is the point where we could
  // see that our "kid" has grown and can make decisions on its own.
  // xTest -> [0.5, 1.5, 2.5, ...]
  // yTest -> [32.9, 34.7, 36.5, ...]
  const xTest = [];
  const yTest = [];
  // By starting from 0.5 and using the same step of 1 as we have used for training set
  // we make sure that test set has different data comparing to training set.
  for (let x = 0.5; x < 100; x += 1) {
    const y = celsiusToFahrenheit(x);
    xTest.push(x);
    yTest.push(y);
  }

  return [xTrain, yTrain, xTest, yTest];
}

// Calculate the cost (the mistake) between the correct output value of 'y' and 'prediction' that NanoNeuron made.
function predictionCost(y, prediction) {
  // This is a simple difference between two values.
  // The closer the values to each other - the smaller the difference.
  // We're using power of 2 here just to get rid of negative numbers
  // so that (1 - 2) ^ 2 would be the same as (2 - 1) ^ 2.
  // Division by 2 is happening just to simplify further backward propagation formula (see below).
  return (y - prediction) ** 2 / 2; // i.e. -> 235.6
}

// Forward propagation.
// This function takes all examples from training sets xTrain and yTrain and calculates
// model predictions for each example from xTrain.
// Along the way it also calculates the prediction cost (average error our NanoNeuron made while predicting).
function forwardPropagation(model, xTrain, yTrain) {
  const m = xTrain.length;
  const predictions = [];
  let cost = 0;
  for (let i = 0; i < m; i += 1) {
    const prediction = nanoNeuron.predict(xTrain[i]);
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
function backwardPropagation(predictions, xTrain, yTrain) {
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

// Train the model.
// This is like a "teacher" for our NanoNeuron model:
// - it will spend some time (epochs) with our yet stupid NanoNeuron model and try to train/teach it,
// - it will use specific "books" (xTrain and yTrain data-sets) for training,
// - it will push our kid to learn harder (faster) by using a learning rate parameter 'alpha'
//   (the harder the push the faster our "nano-kid" will learn but if the teacher will push too hard
//    the "kid" will have a nervous breakdown and won't be able to learn anything).
function trainModel({ model, epochs, alpha, xTrain, yTrain }) {
  // The is the history array of how NanoNeuron learns.
  // It might have a good or bad "marks" (costs) during the learning process.
  const costHistory = [];

  // Let's start counting epochs.
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    // Forward propagation for all training examples.
    // Let's save the cost for current iteration.
    // This will help us to analyse how our model learns.
    const [predictions, cost] = forwardPropagation(model, xTrain, yTrain);
    costHistory.push(cost);

    // Backward propagation. Let's learn some lessons from the mistakes.
    // This function returns smalls steps we need to take for params 'w' and 'b'
    // to make predictions more accurate.
    const [dW, dB] = backwardPropagation(predictions, xTrain, yTrain);

    // Adjust our NanoNeuron parameters to increase accuracy of our model predictions.
    nanoNeuron.w += alpha * dW;
    nanoNeuron.b += alpha * dB;
  }

  // Let's return cost history from the function to be able to log or to plot it after training.
  return costHistory;
}

function getData() {
  return Array.from({ length: 10000 }, (_, index) => {
    const celsius = index + 1;
    return {
      celsius,
      fahrenheit: сelsiusToFahrenheit(celsius),
    };
  });
}

class SimpleNetwork {
  // which is made of single neuron
  constructor() {
    this._accuracy = 0.1; // diff in degrees
    this._weights = [
      random(1, 10),
      //random(1, 10)
      32,
    ];
    this._data = getData();
  }

  train(maxEpoch = 1000) {
    const accuracy = {
      value: Number.MAX_SAFE_INTEGER,
      prev: Number.MAX_SAFE_INTEGER,
      set(value) {
        this.prev = this.value;
        this.value = value;
      },
    };
    const weightAdjustments = [10, 10];
    let i = 0;

    do {
      this._runEpoch(i, accuracy, weightAdjustments);
      i++;

      const { celsius, fahrenheit } = this._data[i - 1];
      console.log({
        epoch: i,
        accuracy: accuracy.value,
        weights: this._weights.join(' '),
        weightAdjustments: weightAdjustments.join(' '),
        celsius,
        fahrenheit,
      });
    } while (i < maxEpoch && accuracy.value > this._accuracy);
  }

  test() {
    console.log('');
    console.info('Testing');

    console.log({
      celsius: 0,
      fahrenheit: this.predict(0),
    });
    console.log({
      celsius: 1,
      fahrenheit: this.predict(1),
    });
  }

  predict(celsius) {
    const [a, b] = this._weights;
    return a * x + b;
  }

  run() {
    this.train();
    this.test();
  }

  _runEpoch(i, accuracy, weightAdjustments) {
    const { celsius, fahrenheit } = this._data[i];
    const predictedResult = this.predict(celsius);

    console.log({ celsius, predictedResult });
    accuracy.set(Math.abs(predictedResult - fahrenheit));

    const isAccuracyLoss = accuracy.value > accuracy.prev;
    if (isAccuracyLoss) {
      const adjustmentCoefficient = 1.15;
      // readjust weight adjustments
      weightAdjustments[0] = weightAdjustments[0] / -adjustmentCoefficient;

      // todo implement proper generic adjustment for each weight
      //   weightAdjustments[1] = weightAdjustments[1] / -adjustmentCoefficient;
    }

    this._weights[0] += weightAdjustments[0];
    // this._weights[1] += weightAdjustments[1]; // todo fix
  }
}

module.exports = SimpleNetwork;
