function idealFormulaCelsiusToFahrenheit(degree) {
  return degree * (9 / 5) + 32;
}

function random(start, end) {
  return Math.random() * (end - start) + start;
}

function getData() {
  return Array.from({ length: 10000 }, (_, index) => {
    const celsius = index + 1;
    return {
      celsius,
      fahrenheit: idealFormulaCelsiusToFahrenheit(celsius),
    };
  });
}

module.exports = class SimpleNetwork {
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
    return this._f(celsius);
  }

  run() {
    this.train();
    this.test();
  }

  _f(x) {
    const [a, b] = this._weights;
    return a * x + b;
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
};
