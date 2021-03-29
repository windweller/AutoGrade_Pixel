const fs = require("fs");

const { getProgram } = require("../@program");

class Color {
  async init(page, body) {
    page.setViewport({
      width: 320,
      height: 450,
    });

    let content = await this.getContent(body);

    await page.setContent(content, {
      waitUntil: ["domcontentloaded"],
    });
  }

  async step(page, { x, y, grid }) {
    if (grid !== undefined) {
      await page.mouse.click(
        (grid % 4) * 80 + 40,
        Math.floor(grid / 4) * 90 + 45
      );
    } else {
      await page.mouse.click(+x, +y);
    }
  }

  async getContent({ script, code }) {
    if (code) {
      script = getProgram(code);
    }

    let content = fs.readFileSync("./src/color/index.html", "utf-8");
    content = content.replace("/// user:script ///", script);

    return content;
  }
}

module.exports = {
  Color,
};
