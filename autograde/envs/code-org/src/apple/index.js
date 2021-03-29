const fs = require("fs");

const { getProgram } = require("../@program");

class Apple {
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
      let { components, matched } = await page.evaluate(() =>
        window.$getComponents()
      );

      if (matched.some((i) => i === +grid)) {
        let apple = components.find((item) => item.id === "apple");
        await page.mouse.click(apple.x, apple.y);
      } else {
        await page.mouse.click(grid % 4 * 80 + 40, Math.floor(grid / 4) * 90 + 45);
      }
    } else {
      await page.mouse.click(+x, +y);
    }
  }

  async getContent({ script, code }) {
    if (code) {
      script = getProgram(code);
    }

    let content = fs.readFileSync("./src/apple/index.html", "utf-8");
    content = content.replace("/// user:script ///", script);
    content = content.replace(
      "/// pic:apple ///",
      `data:image/pmg;base64,${fs.readFileSync(
        "./src/apple/apple.png",
        "base64"
      )}`
    );
    content = content.replace(
      "/// pic:bg ///",
      `data:image/pmg;base64,${fs.readFileSync("./src/apple/bg.png", "base64")}`
    );

    return content;
  }
}

module.exports = {
  Apple,
};
