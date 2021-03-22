const fs = require('fs').promises;
const path = require('path');
const mobilenet = require('@tensorflow-models/mobilenet');
const tfnode = require('@tensorflow/tfjs-node');
const jimp = require('jimp');
const sizeOf = require('image-size');

const readImage = async (filePath) => {
  console.log('Loading and decoding image...');
  await jimp.read(filePath).then((lenna) => {
    const dimensions = sizeOf(filePath);
    const side = 3136;
    return new Promise((resolve, reject) => {
      lenna
        .resize(side, side / (dimensions.width / dimensions.height))
        .write(filePath + 'resized.png', (err) => {
          if (err) {
            reject(err);
            return;
          }

          resolve();
        });
    });
  });
  const imageBuffer = await fs.readFile(filePath + 'resized.png');

  console.log('imageBuffer:');
  console.log(imageBuffer.length);
  //Given the encoded bytes of an image,
  //it returns a 3D or 4D tensor of the decoded image. Supports BMP, GIF, JPEG and PNG formats.
  const tfImage = tfnode.node.decodeImage(imageBuffer);
  return tfImage;
};

const imageClassification = async (imagePath) => {
  const image = await readImage(imagePath);

  console.log('Loading mobilenet...');
  const mobilenetModel = await mobilenet.load();

  console.log('Image classification...');
  const predictions = await mobilenetModel.classify(image);

  console.log('Classification Results:', predictions);
};

// @ts-ignore
const isMainModule = typeof require !== 'undefined' && require.main === module;

if (isMainModule) {
  if (process.argv.length !== 3)
    throw new Error('Incorrect arguments: node classify.js <IMAGE_FILE>');

  imageClassification(process.argv[2]);
}

module.exports = {
  run() {
    console.log('Image classification running...');
    imageClassification(
      process.argv[2] || path.resolve(__dirname, './snail.png'),
    ).catch(console.error);
  },
};
