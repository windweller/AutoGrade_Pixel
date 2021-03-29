const puppeteer = require("puppeteer");

class Browser {
  browser = new Promise(() => {});
  pageMap = new Map();

  constructor({ headless }) {
    this.browser = puppeteer.launch({ headless });
  }

  async init(id, process = 1, force = false) {
    if (this.pageMap.has(id) && !force) {
      return;
    }

    let pages = await Promise.all(
      Array(+process)
        .fill(undefined)
        .map(() => this.initPage())
    );

    this.pageMap.set(id, pages);

    return pages;
  }

  async initPage() {
    return Promise.resolve(this.browser).then((browser) => browser.newPage());
  }

  async reset(id, indexList) {
    if (!this.pageMap.has(id)) {
      return;
    }

    let pages;

    if (indexList?.length) {
      pages = await this.resetPages(this.pageMap.get(id), indexList);
    } else {
      let closedLength = await this.close(id);
      pages = await this.init(id, closedLength, true);
    }

    this.pageMap.set(id, pages);

    return pages;
  }

  async resetPages(pages, indexList) {
    let indexSet = new Set(indexList);
    return Promise.all(
      pages.map(async (page, index) => {
        if (!indexSet.has(index)) {
          return page;
        }

        await page.close();

        return this.initPage();
      })
    );
  }

  async close(id) {
    let oldPages = this.pageMap.get(id);

    for (let page of oldPages) {
      await page.close();
    }

    this.pageMap.delete(id);

    return oldPages.length;
  }

  async getPages(id) {
    return this.pageMap.get(id);
  }
}

module.exports = {
  Browser,
};
