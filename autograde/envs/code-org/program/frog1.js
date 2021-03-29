//GAME SETUP
// Create the sprites
// set velocity for the obstacle and the target

//create the variables
var score = 0;
var health = 100;
var frog = createSprite(176, 325, 104, 70);
frog.setAnimation("frog");


var fly = createSprite(400, randomNumber(100, 260), 67, 47);
fly.setAnimation("fly");
fly.velocityX = -4.5;

var mushroom = createSprite(400, 325, 60, 63);
mushroom.setAnimation("mushroom");
mushroom.velocityX = -3.5;

function draw() {
  // BACKGROUND
  // draw the ground and other background
  background(135, 206, 235);

  // SPRITE INTERACTIONS
  // if the player touches the obstacle
  // the health goes down, and the obstacle turns
  if (frog.isTouching(mushroom)) {
    health -= 1;
    mushroom.rotation = 30;
  } else {
    mushroom.rotation = 0;
  }

  // if the frog touches the fly
  // the score goes up, the fly resets

  if (frog.isTouching(fly)) {
    score += 1;

    fly.x = 400;
    fly.y = randomNumber(130, 260);
  }

  // JUMPING
  // if the player has reached the ground
  // stop moving down
  if (frog.y >= 325) {
    frog.velocityY = 0;
  }

  // if the player presses the up arrow
  // start moving up
  if (keyWentDown("up") && frog.velocityY === 0) {
    frog.velocityY = -4;
  }

  // if the player reaches the top of the jump
  // start moving down
  if (frog.y <= 135) {
    frog.velocityY = 4;
  }

  // LOOPING
  // if the obstacle has gone off the left hand side of the screen,
  // move it to the right hand side of the screen

  if (mushroom.x <= 0) {
    mushroom.x = 400;
  }

  // if the target has gone off the left hand side of the screen,
  // move it to the right hand side of the screen

  if (fly.x <= 0) {
    fly.x = 400;
    fly.y = randomNumber(130, 260);
  }

  // DRAW SPRITES
  drawSprites();

  // SCOREBOARD
  // add scoreboard and health meter
  fill("black");
  textSize(20);
  text("Health:", 280, 30);
  text(health, 350, 30);

  text("Score:", 30, 30);
  text(score, 100, 30);
  // GAME OVER
  // if health runs out
  // show Game over
  if (health < 0) {
    background("black");
    fill("green");
    textSize(50);
    text("Game Over!", 40, 200);
  }
}
