var init = false;
var backgroundStyle = hardcourt_bg;
var paddleSpeed = 6;
var ballStyle = "hardcourt_ball";
var ballSpeed = 2;
var paddle = createSprite(122, 372, 44, 18);

paddle.setAnimation("hardcourt_paddle");

var goal = createSprite(200, 0, 200, 1);
var wallLeft = createSprite(5, 200, 10, 400);
var wallRight = createSprite(395, 200, 10, 400);
var wallTopLeft = createSprite(50, 5, 100, 10);
var wallTopRight = createSprite(350, 5, 100, 10);

goal.visible = false;
wallLeft.visible = false;
wallRight.visible = false;
wallTopLeft.visible = false;
wallTopRight.visible = false;

var balls = new Group();

var fnDict = {
  "move left": function () {
    if (paddle.isTouching(wallLeft)) {
      return;
    }

    paddle.x -= paddleSpeed;
  },
  "move right": function () {
    if (paddle.isTouching(wallRight)) {
      return;
    }

    paddle.x += paddleSpeed;
  },
  "bounce ball": function (target, ball) {
    if (!target || !ball) {
      return;
    }

    ball.bounceOff(target);
  },
  "score point": () => (score_1 += 1),
  "score opponent point": () => (score_2 += 1),
  "launch new ball": () => {
    let x = randomNumber(48, 352);
    let ball = createSprite(x, randomNumber(140, 160), 28, 28);
    ball.setAnimation(ballStyle);
    ball.velocityY = ballSpeed;
    let r = Math.random() * 4 - 2;
    ball.velocityX = x < 100 ? Math.abs(r) : x > 300 ? -Math.abs(r) : r;

    balls.add(ball);
  },
  "set 'random' paddle speed": () => setPaddleSpeed("random"),
  "set 'very slow' paddle speed": () => setPaddleSpeed("very slow"),
  "set 'slow' paddle speed": () => setPaddleSpeed("slow"),
  "set 'normal' paddle speed": () => setPaddleSpeed("normal"),
  "set 'fast' paddle speed": () => setPaddleSpeed("fast"),
  "set 'very fast' paddle speed": () => setPaddleSpeed("very fast"),

  "set 'random' ball speed": () => setBallSpeed("random"),
  "set 'very slow' ball speed": () => setBallSpeed("very slow"),
  "set 'slow' ball speed": () => setBallSpeed("slow"),
  "set 'normal' ball speed": () => setBallSpeed("normal"),
  "set 'fast' ball speed": () => setBallSpeed("fast"),
  "set 'very fast' ball speed": () => setBallSpeed("very fast"),
  "set 'random' scene": () =>
    (backgroundStyle = randomNumber(0, 1) ? hardcourt_bg : retro_bg),
  "set 'hardcourt' scene": () => (backgroundStyle = hardcourt_bg),
  "set 'retro' scene": () => (backgroundStyle = retro_bg),
  "set 'random' scene": () =>
    (backgroundStyle = randomNumber(0, 1) ? hardcourt_bg : retro_bg),
  "set 'hardcourt' scene": () => (backgroundStyle = hardcourt_bg),
  "set 'retro' scene": () => (backgroundStyle = retro_bg),
  "set 'random' ball": () => (
    (ballStyle = randomNumber(0, 1) ? "hardcourt_ball" : "retro_ball"),
    balls.toArray().forEach((ball) => ball.setAnimation(ballStyle))
  ),
  "set 'hardcourt' ball": () => (
    (ballStyle = "hardcourt_ball"),
    balls.toArray().forEach((ball) => ball.setAnimation(ballStyle))
  ),
  "set 'retro' ball": () => (
    (ballStyle = "retro_ball"),
    balls.toArray().forEach((ball) => ball.setAnimation(ballStyle))
  ),
  "set 'random' paddle": () =>
    paddle.setAnimation(
      randomNumber(0, 1) ? "hardcourt_paddle" : "retro_paddle"
    ),
  "set 'hardcourt' paddle": () => paddle.setAnimation("hardcourt_paddle"),
  "set 'retro' paddle": () => paddle.setAnimation("retro_paddle"),
};

window.setPaddleSpeed = (level) => {
  switch (level) {
    case "random":
      paddleSpeed = Math.random() * 10;
      break;
    case "very slow":
      paddleSpeed = 2;
      break;
    case "slow":
      paddleSpeed = 3;
      break;
    case "normal":
      paddleSpeed = 6;
      break;
    case "fast":
      paddleSpeed = 8;
      break;
    case "very fast":
      paddleSpeed = 12;
      break;
  }
};

window.setBallSpeed = (level) => {
  switch (level) {
    case "random":
      ballSpeed = [1, 1.4, 2, 3.6, 6][randomNumber(0, 4)];
      break;
    case "very slow":
      ballSpeed = 1;
      break;
    case "slow":
      ballSpeed = 1.4;
      break;
    case "normal":
      ballSpeed = 2;
      break;
    case "fast":
      ballSpeed = 3.6;
      break;
    case "very fast":
      ballSpeed = 6;
      break;
  }

  for (const ball of balls) {
    ball.velocityY = (ball.velocityY / Math.abs(ball.velocityY)) * ballSpeed;
  }
};

window.runHooks = (name, ...args) => {
  for (let action of $data[name] || []) {
    if (fnDict[action]) {
      fnDict[action](...args);
    }
  }
};

function draw() {
  background(backgroundStyle);

  if (!init) {
    init = true;
    runHooks("when run");
  }

  if (keyDown("left")) {
    runHooks("when left arrow");
  }

  if (keyDown("right")) {
    runHooks("when right arrow");
  }

  for (const ball of balls) {
    if (ball.isTouching(paddle)) {
      runHooks("when ball hits paddle", paddle, ball);
    }

    if (ball.isTouching(goal)) {
      if (ball.willRemove) {
        continue;
      }

      runHooks("when ball in goal", goal, ball);

      ball.willRemove = setTimeout(() => {
        ball.remove();
      }, 1000);
    }

    for (const wall of [wallLeft, wallRight, wallTopLeft, wallTopRight]) {
      if (ball.isTouching(wall)) {
        runHooks("when ball hits wall", wall, ball);
      }
    }

    if (ball.y > 400) {
      runHooks("when ball misses paddle");
      ball.remove();
    }

    if (ball.x < -48 || ball.x > 448 || ball.y < -48) {
      if (ball.willRemove) {
        continue;
      }

      ball.remove();
    }
  }

  // DRAW SPRITES
  drawSprites();

  strokeWeight(4);
  stroke("#000");
  fill("#FFF");
  textSize(32);
  let t = "Score : " + score_1 + " : " + score_2;
  text(t, 200 - textWidth(t) / 2, 64);
  
}
