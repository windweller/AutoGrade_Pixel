const Koa = require("koa");
const Router = require("koa-router");
const bodyParser = require("koa-bodyparser");

class Net {
  app;

  constructor({ port, fns }) {
    let app = new Koa();
    let router = new Router();

    router.all("/:type/:id", async (ctx) => {
      let {
        params: { type, id },
        request: { body },
      } = ctx;

      // init reset step close
      if (!fns?.[type]) {
        return;
      }

      console.log(`${type} ${id}...`);

      ctx.body = await fns[type](id, body);
    });

    app
      .use(bodyParser())
      .use(router.routes())
      .listen(port, () => console.log(`Sever listening on ${port}`));

    this.app = app;
  }
}

module.exports = {
  Net,
};
