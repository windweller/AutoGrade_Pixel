const fs = require("fs");

const { getProgram } = require("../@program");

class Bounce {
  async init(page, body) {
    page.setViewport({
      width: 400,
      height: 400,
    });

    let content = await this.getContent(body);

    await page.setContent(content, {
      waitUntil: ["networkidle0"],
    });
  }

  async step(page, { key }) {
    if (key) {
      for (let item of key.split("|")) {
        if(!item) {
          continue
        }
        
        page.keyboard.down(item);
      }
    }
    
    await page.evaluate(() => window.$runFrames());

    if (key) {
      for (let item of key.split("|")) {
        if(!item) {
          continue
        }
        
        page.keyboard.up(item);
      }
    }
  }

  async getContent({ script, code, frames }) {
    if (code) {
      script = getProgram(code);
    }

    let content = fs.readFileSync("./src/bounce/index.html", "utf-8");
    let wrapper = fs.readFileSync("./src/bounce/wrapper.js", "utf-8");
    content = content.replace("/// wrapper:script ///", wrapper);
    content = content.replace("/// user:script ///", script);

    if (frames) {
      content = content.replace(
        "window.$defaultFrames = Infinity",
        `window.$defaultFrames = ${frames}`
      );
      content = content.replace(
        "window.$frames = $defaultFrames;",
        "window.$frames = 1;"
      );
    }

    return content;
  }
}

module.exports = {
  Bounce,
};
