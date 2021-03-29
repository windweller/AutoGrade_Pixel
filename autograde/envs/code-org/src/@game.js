const { Browser } = require("./@browser");
const { Net } = require("./@net");

class Game {
  browser;
  net;
  initParamsMap = new Map();

  constructor({ headless, port, game }) {
    this.game = game;
    this.browser = new Browser({ headless });
    this.net = new Net({
      port,
      fns: {
        init: this.init,
        step: this.step,
        reset: this.reset,
        close: this.close,
        user_interface: this.userInterface,
      },
    });
  }

  /**
   * @param {*} id
   * @param {
   *  process,
   *  script,
   *  code,
   *  format,
   *  quality
   * }
   */
  init = async (id, body) => {
    let pages = await this.browser.init(id, body.process);

    if (!pages) {
      return;
    }

    this.initParamsMap.set(id, body);

    return Promise.all(
      pages.map((page) => this._handler("init", id, page, body))
    );
  };

  reset = async (id, body) => {
    let initParams = this.initParamsMap.get(id);

    if (!initParams) {
      return;
    }

    let pages = await this.browser.reset(id, body?.actions);

    if (!pages) {
      return;
    }

    let indexSet = new Set(
      (body?.actions?.length ? body.actions : Object.keys(pages)).map((i) =>
        Number(i)
      )
    );

    await Promise.all(
      pages.map((page, index) =>
        indexSet.has(index)
          ? this._handler("init", id, page, initParams)
          : undefined
      )
    );

    return Promise.all(
      pages.map((page) => getResponse(page, initParams.quality))
    );
  };

  step = async (id, body) => {
    let pages = await this.browser.getPages(id);

    return Promise.all(
      pages.map((page, index) => {
        let data = body?.actions?.[index];

        if (!data) {
          return undefined;
        }

        return this._handler("step", id, page, data);
      })
    );
  };

  close = async (id) => {
    await this.browser.close(id);
    this.initParamsMap.delete(id);
  };

  userInterface = async (code) => {
    return this.game.getContent({ code });
  };

  async _handler(type, id, page, body) {
    if (type === "init") {
      setFormat(page, body.format);
    }

    await this.game[type]?.(page, body);

    let initParams = this.initParamsMap.get(id);

    if (!initParams) {
      return;
    }

    return getResponse(page, initParams.quality);
  }
}

async function getResponse(page, quality) {
  return {
    ...(containFormat(page, "img")
      ? {
          // data:image/jpeg;base64,
          img: (
            await page.screenshot({
              type: "jpeg",
              quality,
            })
          ).toString("base64"),
        }
      : {}),
    ...(containFormat(page, "state")
      ? {
          state: await page.evaluate(() => window.$getComponents()),
        }
      : {}),
  };
}

function setFormat(page, formatString = "img|state") {
  page.$fmt = formatString
    .split("|")
    .reduce((obj, flag) => ((obj[flag] = true), obj), {});
}

function containFormat(page, flag) {
  return !!(page.$fmt && page.$fmt[flag]);
}

module.exports = {
  Game,
};
