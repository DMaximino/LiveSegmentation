const webcam = new Webcam(document.getElementById('wc'));
let isPredicting = false;
let deeplab_model;
const model = 'pascal';
const colormap = deeplab.getColormap(model);
const labels = deeplab.getLabels(model);
const config = {canvas: output, colormap: colormap, labels: labels }

const loadModel = async () => {
  const base = 'pascal';        // set to your preferred model, out of `pascal`,
                                // `cityscapes` and `ade20k`
  const quantizationBytes = 4;  // either 1, 2 or 4
  // use the getURL utility function to get the URL to the pre-trained weights
  const modelUrl = deeplab.getURL(base, quantizationBytes);
  const rawModel = await tf.loadGraphModel(modelUrl);
  const modelName = 'pascal';  // set to your preferred model, out of `pascal`,
  // `cityscapes` and `ade20k`
  deeplab_model = new deeplab.SemanticSegmentation(rawModel);
  return deeplab_model;
};

async function predict() {
  while (isPredicting) {
  tf.tidy(() => {
  const img = webcam.capture();
  deeplab_model.segment(img, config);
  })

  await tf.nextFrame();
  }
}

function tooglePredicting(){
  isPredicting = !isPredicting;
  var elem = document.getElementById("tooglePredictingButton");
  if(isPredicting)
  {
    predict();
    elem.value = "Stop"
  }
  else
  {
    elem.value = "Start"
  }
}

function stopPredicting(){
	isPredicting = false;
}

async function init(){
	await webcam.setup();
  loadModel().then(document.getElementById("tooglePredictingButton").disabled = false);
}


init();