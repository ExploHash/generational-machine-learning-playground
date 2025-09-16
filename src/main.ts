import sdl from '@kmamal/sdl'
import { createCanvas } from '@napi-rs/canvas'
import { setTimeout } from 'timers/promises'
// Setup
const window = sdl.video.createWindow({ title: "Canvas2D", width: 800, height: 400 })
const { pixelWidth: width, pixelHeight: height } = window
const canvas = createCanvas(width, height)
const ctx = canvas.getContext('2d')
// import * as tf from '@tensorflow/tfjs-node'
import * as tf from '@tensorflow/tfjs-node-gpu'



type Blob = {
  model: tf.LayersModel;
  orientation: number;
  x: number;
  y: number;
  size: number;
  color: string;
  fitness: number;
  gas: number;
  uninteruptedSteeringDegree: number;
  lastTurnDirection: 'left' | 'right' | 'straight';
  amountOfActions: number;
  amountOfOversteeringActions: number;
  amountOfDistanceTraveled: number;
  amountOfObstaclesHit: number;
  currentlyHittingObstacle: boolean;
}

type Obsctacle = {
  x: number;
  y: number;
  width: number;
  height: number;
}

ctx.fillStyle = 'blue'
ctx.fillRect(0, 0, width, height)


const listOfColors = [
  'red', 'green', 'blue', 'yellow', 'purple', 'orange'
]
const populationSize = 25;

let visibleBlobs: Blob[] = [];

const obstacles: Obsctacle[] = [
  { x: 200, y: 100, width: 50, height: 200 },
  { x: 400, y: 200, width: 50, height: 200 },
  { x: 600, y: 0, width: 50, height: 200 },
];


const finishLineX = width - 50;

function createModel() {
  const model = tf.sequential();
  const inputs = 8; // x, y, orientation, distance to finish line
  model.add(tf.layers.dense({inputShape: [inputs], units: 8, activation: 'relu'}));
  model.add(tf.layers.dense({units: 3, activation: 'tanh'})); // left, right
  return model;
}

function initializeBlob(model: tf.LayersModel, color: string): Blob {
  const size = 20;

  return {
    model,
    currentlyHittingObstacle: false,
    orientation: 90, // Facing right
    uninteruptedSteeringDegree: 0,
    gas: 0,
    lastTurnDirection: 'straight',
    amountOfActions: 0,
    amountOfOversteeringActions: 0,
    amountOfDistanceTraveled: 0,
    amountOfObstaclesHit: 0,
    x: size * 2,
    y: height / 2 - size / 2,
    size,
    color,
    fitness: 0
  }
}

function resetBlob(blob: Blob) {
  blob.x = blob.size * 2;
  blob.y = height / 2 - blob.size / 2;
  blob.orientation = 90;
  blob.fitness = 0;
  blob.uninteruptedSteeringDegree = 0;
  blob.gas = 0;
  blob.lastTurnDirection = 'straight';
  blob.amountOfActions = 0;
  blob.amountOfOversteeringActions = 0;
  blob.amountOfDistanceTraveled = 0;
  blob.amountOfObstaclesHit = 0;
  blob.currentlyHittingObstacle = false;
}

// function mutateModel(model: tf.LayersModel, mutationRate: number = 0.1) {
//   const weights = model.getWeights();
//   const newWeights = weights.map(weight => {
//     const shape = weight.shape;
//     const values = weight.arraySync() as number[] | number[][];
//     const mutatedValues = (Array.isArray(values[0]) ? values as number[][] : [values as number[]]).map(row =>
//       row.map(value => (Math.random() < mutationRate ? value + tf.randomNormal([1]).arraySync()[0] : value))
//     );
//     return tf.tensor(Array.isArray(values[0]) ? mutatedValues : mutatedValues[0], shape);
//   });
//   model.setWeights(newWeights);
// }


function mutateModel(model: tf.LayersModel, mutationRate = 0.1, mutationStrength = 0.1) {
  const weights = model.getWeights();
  const newWeights = weights.map(tensor => {
    const vals = tensor.dataSync().slice();
    for (let i = 0; i < vals.length; i++) {
      if (Math.random() < mutationRate) {
        vals[i] += (Math.random() * 2 - 1) * mutationStrength;
      }
    }
    return tf.tensor(vals, tensor.shape);
  });
  model.setWeights(newWeights);
}

function cloneModel(model: tf.LayersModel): tf.LayersModel {
  const newModel = createModel();
  newModel.setWeights(model.getWeights());
  return newModel;
}

function evolveBlobs(blobs: Blob[]): Blob[] {
  // Fitness is already calculated during simulation, no need to recalculate

  // Sort blobs by fitness (higher is better)
  blobs.sort((a, b) => b.fitness - a.fitness);

  // Select the top 20%
  const eliteCount = Math.max(1, Math.floor(blobs.length * 0.1));
  const elites = blobs.slice(0, eliteCount);

  // Create new blobs through mutation
  const newPop = [...elites];

  // Reset positions and orientations of elites for the new generation
  for (const elite of newPop) {
    resetBlob(elite);
  }

  // Fill the rest of the population with mutated offspring
  while (newPop.length < populationSize) {
    const parent = elites[Math.floor(Math.random() * elites.length)];
    const childModel = cloneModel(parent.model);
    mutateModel(childModel);
    newPop.push(initializeBlob(childModel, listOfColors[newPop.length % listOfColors.length]));
  }

  return newPop;
}

function renderBlobs() {
  for (const blob of visibleBlobs) {
    ctx.fillStyle = blob.color
    ctx.beginPath()
    ctx.arc(blob.x, blob.y, blob.size, 0, Math.PI * 2)
    ctx.fill()
  }
}

function renderObstacles() {
  for (const obstacle of obstacles) {
    ctx.fillStyle = 'gray';
    ctx.fillRect(obstacle.x, obstacle.y, obstacle.width, obstacle.height);
  }
}

function calculateFitness(blob: Blob): number {
  const distanceToFinishLine = finishLineX - blob.x;
  let fitness = Math.max(0, width - distanceToFinishLine); // Higher is better

  // Penalty for steering longer than 110 degrees in one direction
  fitness -= blob.amountOfOversteeringActions * 0.2;

  // Penalty for obstacles hit (not implemented in current logic, placeholder)
  fitness -= blob.amountOfObstaclesHit * 0.5;

  // Bonus for distance traveled
  // fitness += blob.amountOfDistanceTraveled * 0.01;

  console.log(`Blob fitness: ${fitness.toFixed(2)})}`);
  return fitness;
}

function setModelRandomWeights(model: tf.LayersModel) {
  const weights = model.getWeights();
  const newWeights = weights.map(weight => {
    const shape = weight.shape;
    const values = tf.randomNormal(shape).arraySync() as number[] | number[][];
    return tf.tensor(values, shape);
  });
  model.setWeights(newWeights);
}


function render(generation: number, second: number = 0) {
  // Fill background with dark blue
  ctx.fillStyle = 'darkblue'
  ctx.fillRect(0, 0, width, height)

  // Draw finish line
  ctx.fillStyle = 'white'
  ctx.fillRect(finishLineX, 0, 10, height)

  // Render generation number
  ctx.fillStyle = 'white'
  ctx.font = '20px Adwaita Sans'
  ctx.fillText(`Gen: ${generation} - Sec: ${second}`, 10, 30)

  // Render blobs
  renderBlobs()

  // Render obstacles
  renderObstacles()

  // Render to window
  const buffer = Buffer.from(ctx.getImageData(0, 0, width, height).data)
  window.render(width, height, width * 4, 'rgba32', buffer)
}

function calculateBlobAntennae(blob: Blob): { leftCorner: number; rightCorner: number, center: number } {
  // Calculate the distance 45 degrees to the left, right, and straight ahead against obstacles and walls
  const antennaeLength = 50; // Length of the antennae

  // Calculate three directions: left (-45°), center (0°), right (+45°)
  const directions = [-45, 0, 45];
  const results = { leftCorner: antennaeLength, rightCorner: antennaeLength, center: antennaeLength };
  const resultKeys = ['leftCorner', 'center', 'rightCorner'];

  for (let i = 0; i < directions.length; i++) {
    const angleOffset = directions[i];
    const checkAngle = blob.orientation + angleOffset;
    const radians = (checkAngle * Math.PI) / 180;

    // Cast ray from blob position in the direction
    for (let distance = 1; distance <= antennaeLength; distance++) {
      const checkX = blob.x + Math.cos(radians) * distance;
      const checkY = blob.y + Math.sin(radians) * distance;

      // Check against world boundaries
      if (checkX <= 0 || checkX >= width || checkY <= 0 || checkY >= height) {
        results[resultKeys[i] as keyof typeof results] = distance;
        break;
      }

      // Check against obstacles
      let hitObstacle = false;
      for (const obstacle of obstacles) {
        if (
          checkX >= obstacle.x &&
          checkX <= obstacle.x + obstacle.width &&
          checkY >= obstacle.y &&
          checkY <= obstacle.y + obstacle.height
        ) {
          results[resultKeys[i] as keyof typeof results] = distance;
          hitObstacle = true;
          break;
        }
      }

      if (hitObstacle) break;
    }
  }

  return results;
}

async function simulate(generation: number = 1) {
  const framesPerSecond = 60;
  const maxSimulationTime = 120; // seconds
  const maxFrames = framesPerSecond * maxSimulationTime;
  let frameCount = 0;

  while (frameCount < maxFrames) {
    // Render all blobs
    // Render every 8 frames to improve performance
    if (frameCount % 8 === 0){
      render(generation, Math.floor(frameCount / framesPerSecond));
    }

    // Update each blob
    updateBlobs()

    // await setTimeout(1000 / framesPerSecond)
    frameCount++;
  }
}

function generateInitialBlobs(): Blob[] {
  const blobs = [];
  for (let i = 0; i < populationSize; i++) {
    const color = listOfColors[i % listOfColors.length];
    const model = createModel();
    setModelRandomWeights(model);
    const blob = initializeBlob(model, color);
    blobs.push(blob);
  }

  return blobs;
}

function updateBlobs() {
  for (const blob of visibleBlobs) {
    // Skip movement if blob has reached the finish line
    if (blob.x + blob.size >= finishLineX) {
      continue;
    }

    // Prepare input tensor: [x, y, orientation, distance to finish line]
    const { leftCorner, rightCorner, center } = calculateBlobAntennae(blob);
    const distanceToFinishLine = finishLineX - blob.x;
    const inputTensor = tf.tensor2d([[blob.x / width, blob.y / height, blob.orientation / 360, distanceToFinishLine / width, leftCorner / 50, rightCorner / 50, center / 50, blob.gas]]); // Normalize inputs
    const outputTensor = blob.model.predict(inputTensor) as tf.Tensor;

    const output = outputTensor.arraySync() as number[][];
    const [turnLeft, turnRight, gas] = output[0];
    blob.gas = Math.max(0, gas); // Ensure gas is non-negative

    // Update orientation based on model output
    if (turnLeft > 0.1) {
      blob.orientation -= 5;
      if (blob.lastTurnDirection === 'left') {
        blob.uninteruptedSteeringDegree += 5;
        if (blob.uninteruptedSteeringDegree > 60) {
          blob.amountOfOversteeringActions++;
          blob.uninteruptedSteeringDegree = 0; // Reset after counting
        }
      } else {
        blob.uninteruptedSteeringDegree = 5;
        blob.lastTurnDirection = 'left';
        blob.amountOfActions++;
      }
    }
    if (turnRight > 0.1) {
      blob.orientation += 5; // Turn right
      if (blob.lastTurnDirection === 'right') {
        blob.uninteruptedSteeringDegree += 5;
        if (blob.uninteruptedSteeringDegree > 90) {
          blob.amountOfOversteeringActions++;
          blob.uninteruptedSteeringDegree = 0; // Reset after counting
        }
      } else {
        blob.uninteruptedSteeringDegree = 5;
        blob.lastTurnDirection = 'right';
        blob.amountOfActions++;
      }
    }

    // Keep orientation within 0-360 degrees
    if (blob.orientation < 0) blob.orientation += 360;
    if (blob.orientation >= 360) blob.orientation -= 360;

    // Move forward in the direction of orientation
    const radians = (blob.orientation * Math.PI) / 180;
    const newX = blob.x + Math.cos(radians) * 2 * gas; // Move speed
    const newY = blob.y + Math.sin(radians) * 2 * gas; // Move speed
    const distanceTraveled = Math.sqrt(Math.pow(newX - blob.x, 2) + Math.pow(newY - blob.y, 2));
    blob.amountOfDistanceTraveled += distanceTraveled;

    // Check if new position is within bounds
    if (newX > blob.size && newX < width - blob.size
      && newY > blob.size && newY < height - blob.size) {
      // Check for obstacle collisions
      let collision = false;
      for (const obstacle of obstacles) {
        if (
          newX + blob.size > obstacle.x &&
          newX - blob.size < obstacle.x + obstacle.width &&
          newY + blob.size > obstacle.y &&
          newY - blob.size < obstacle.y + obstacle.height
        ) {
          collision = true;
          break;
        }
      }
      if (!collision) {
        blob.currentlyHittingObstacle = false;
        blob.x = newX;
        blob.y = newY;
      // } else if (!blob.currentlyHittingObstacle) {
      } else {
        // Penalty for hitting an obstacle
        blob.currentlyHittingObstacle = true;
        blob.amountOfObstaclesHit = (blob.amountOfObstaclesHit || 0) + 1;
      }
    } else {
      blob.currentlyHittingObstacle = true;
      // Penalty for hitting wall
      blob.amountOfObstaclesHit = (blob.amountOfObstaclesHit || 0) + 1;
    }

    // Clean up tensors
    inputTensor.dispose();
    outputTensor.dispose();

    // Update fitness to current position (not just best position)
    blob.fitness = calculateFitness(blob);
  }
}

async function runWorld() {
  const maxGenerations = 100;

  for (let generation = 1; generation <= maxGenerations; generation++) {
    console.log(`Generation ${generation}`);
    // Simulate current generation
    await simulate(generation);

    // Evolve to next generation
    console.log(visibleBlobs)
    visibleBlobs = evolveBlobs(visibleBlobs);
  }
}

async function initialize () {
  visibleBlobs = generateInitialBlobs()
  await runWorld()
}

initialize()
