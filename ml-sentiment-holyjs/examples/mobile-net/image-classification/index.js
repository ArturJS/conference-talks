const tfnode = require('@tensorflow/tfjs-node');
const mobilenet = require('@tensorflow-models/mobilenet');
const fs = require('fs').promises;
const path = require('path');

const classify = async (imagePath) => {
  const image = await fs.readFile(imagePath);
  const decodedImage = tfnode.node.decodeImage(image, 3);
  const model = await mobilenet.load();
  const predictions = await model.classify(decodedImage);
  console.log('predictions:', predictions);
};

// @ts-ignore
const isMainModule = typeof require !== 'undefined' && require.main === module;

if (isMainModule) {
  if (process.argv.length !== 3)
    throw new Error('Incorrect arguments: node classify.js <IMAGE_FILE_PATH>');

  classify(process.argv[2]);
}

module.exports = {
  run() {
    const imagePath = path.join(__dirname, './snail.png');
    classify(imagePath).catch(console.error);
  },
};

// see also https://gist.github.com/jthomas/145610bdeda2638d94fab9a397eb1f1d#gistcomment-3416190
