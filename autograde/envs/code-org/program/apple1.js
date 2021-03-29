var score = 0;
var lives = 3;
onEvent("apple", "click", function () {
  score = score + 1;
  setText("score_label", score);
  setPosition("apple", randomNumber(50, 280), randomNumber(50, 350));
});

onEvent("background", "click", function () {
  lives = lives - 1;
  setText("lives_label", lives);
  setPosition("apple", randomNumber(50, 280), randomNumber(50, 350));
});
