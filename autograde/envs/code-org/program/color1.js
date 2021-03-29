var current = 1;
var bug;

function setPlayer() {
  if (current == 1) {
    setProperty(
      "player1_highlight",
      "background-color",
      rgb(197, 197, 197, 0.408)
    );
    setProperty(
      "player2_highlight",
      "background-color",
      rgb(197, 197, 197, 0)
    );
  } else {
    setProperty(
      "player2_highlight",
      "background-color",
      rgb(197, 197, 197, 0.408)
    );
    setProperty(
      "player1_highlight",
      "background-color",
      rgb(197, 197, 197, 0)
    );
  }
}

function setColor(init) {
  var r = randomNumber(0, 255);
  var g = randomNumber(0, 255);
  var b = randomNumber(0, 255);
  var color1 = rgb(r, g, b, 1);
  var color2 = rgb(r, g, b, 0.5);

  bug = randomNumber(1, 4);

  setProperty("button1", "background-color", color1);
  setProperty("button2", "background-color", color1);
  setProperty("button3", "background-color", color1);
  setProperty("button4", "background-color", color1);
  setProperty("button" + bug, "background-color", color2);
}

setColor();
setPlayer();

function onEnt(id) {
  onEvent("button" + id, "click", function () {
    var score = getNumber("score" + current + "_label");

    if (bug == id) {
      score = score + 1;
    } else {
      score = score - 3;
    }

    setNumber("score" + current + "_label", score);

    if (current == 1) {
      current = 2;
    } else {
      current = 1;
    }

    setColor();
    setPlayer();
  });
}

onEnt(1);
onEnt(2);
onEnt(3);
onEnt(4);