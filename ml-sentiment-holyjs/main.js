const { Select } = require('enquirer');
const simpleNeuralNetwork = require('./examples/simple-neural-network');
const imageClassification = require('./examples/mobile-net/image-classification');

function start() {
  const exampleNeuralNetworks = {
    'simple-neural-network': simpleNeuralNetwork,
    'mobile-net\\image-classification': imageClassification,
    'sentiment-ai': {
      run() {
        throw new Error('Error: ToDo Implement :-/');
      },
    },
  };

  const prompt = new Select({
    name: 'color',
    message: 'Pick an example:',
    choices: Object.keys(exampleNeuralNetworks),
  });

  prompt
    .run()
    .then((answer) => {
      return exampleNeuralNetworks[answer].run();
    })
    .catch(console.error);
}

start();
