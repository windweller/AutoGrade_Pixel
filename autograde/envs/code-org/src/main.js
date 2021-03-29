const { program } = require("commander");
const { Game } = require("./@game");

program
  .version("0.0.1")
  .option("-p, --port <type>", "socket port", (num) => parseInt(num), 3300)
  .option("-o, --open", "close handless mode, open chrome")
  .option("-g, --game <name>", "game type", "apple")
  .parse(process.argv);

if (program.game === "apple") {
  const { Apple } = require("./apple");
  starGame(new Apple());
} else if (program.game === "color") {
  const { Color } = require("./color");
  starGame(new Color());
}else if (program.game === "frog") {
  const { Frog } = require("./frog");
  starGame(new Frog());
}else if (program.game === "flyer") {
  const { Flyer } = require("./flyer");
  starGame(new Flyer());
}else if (program.game === "bounce") {
  const { Bounce } = require("./bounce");
  starGame(new Bounce());
}

function starGame(game) {
  const headless = !program.open;
  const port = program.port;
  new Game({ headless, port, game });
}
