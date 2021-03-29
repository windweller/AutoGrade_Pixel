const fs = require("fs");
const path = require("path");

function joinProgram(file) {
  return path.join(__dirname, "../program", file);
}

function getProgram(file) {
  let src = joinProgram(file);

  if (!fs.existsSync(src)) {
    src = joinProgram(`${file}.js`);
  } 

  if (!fs.existsSync(src)) {
    src = joinProgram(`${file}.json`);
  }

  return fs.readFileSync(src, "utf-8");
}

module.exports = {
  getProgram,
};
