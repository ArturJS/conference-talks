const { Select } = require('enquirer');
const simpleNetwork = require('./examples/simple-network');

const exampleNeuralNetworks = {
  'simple-network': simpleNetwork,
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
