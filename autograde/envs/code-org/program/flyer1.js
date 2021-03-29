// GAME SETUP
// create player, target, and obstacles
var player = createSprite(200, 100);
player.setAnimation("fly_bot");
player.scale = 0.8;
player.velocityY = 4;

var coin = createSprite(randomNumber(50, 350), randomNumber(50, 350));
coin.setAnimation("coin");

var rockX = createSprite(0, randomNumber(50, 350));
rockX.setAnimation("rock");
rockX.velocityX = 4;

var rockY = createSprite(randomNumber(50, 350), 0);
rockY.setAnimation("rock");
rockY.velocityY = 4;

function draw() {
  background("lightblue");

  // FALLING

  // LOOPING

  if (rockX.x > 450) {
    rockX.x = -50;
    rockX.y = randomNumber(50, 350);
  }

  if (rockY.y > 450) {
    rockY.y = -50;
    rockY.x = randomNumber(50, 350);
  }

  // PLAYER CONTROLS
  // change the y velocity when the user clicks "up"
  if (keyDown("up")) {
    player.velocityY = -10;
  } else {
    if (player.velocityY <= 10) {
      player.velocityY += 1;
    } else {
      player.velocityY = 10;
    }
  }

  // decrease the x velocity when user clicks "left"

  if (keyDown("left")) {
    if (player.velocityX > -2) {
      player.velocityX -= 0.2;
    } else {
      player.velocityX = -2;
    }
  }

  // increase the x velocity when the user clicks "right"
  if (keyDown("right")) {
    if (player.velocityX < -2) {
      player.velocityX += 0.2;
    } else {
      player.velocityX = 2;
    }
  }

  // SPRITE INTERACTIONS
  // reset the coin when the player touches it
  if (player.isTouching(coin)) {
    coin.x = randomNumber(50, 350);
    coin.y = randomNumber(50, 350);
  }

  // make the obstacles push the player

  if (rockX.isTouching(player)) {
    rockX.displace(player);
  }

  if (rockY.isTouching(player)) {
    rockY.displace(player);
  }

  // DRAW SPRITES
  drawSprites();

  // GAME OVER
  if (player.x < -50 || player.x > 450 || player.y < -50 || player.y > 450) {
    background("black");
    textSize(50);
    fill("green");
    text("Game Over!", 50, 200);
  }
}
