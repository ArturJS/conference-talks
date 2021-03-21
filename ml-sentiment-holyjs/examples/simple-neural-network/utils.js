// @ts-check

function celsiusToFahrenheit(celsius) {
  return 1.8 * celsius + 32;
}

module.exports = {
  celsiusToFahrenheit,

  generateDataSets() {
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
  },

  predictionCost(y, prediction) {
    return (y - prediction) ** 2 / 2;
  },

  random(start, end) {
    return Math.random() * (end - start) + start;
  },
};
