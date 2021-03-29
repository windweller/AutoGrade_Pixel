# OpenAI gym wrapper around node.js games environment server

## Environments

Here are the supporeted code.org interactive code assignments:

Flyer game: https://studio.code.org/s/csd3-2020/stage/23/puzzle/2 

Frog jump: https://studio.code.org/s/csd3-2020/stage/20/puzzle/2 

Color Sleuth: https://studio.code.org/s/csp5-virtual/stage/10/puzzle/1 

Apple Grab: https://studio.code.org/s/csp5-virtual/stage/5/puzzle/1 

### Requirements
* `python==3.7.x`
* `tensorflow==1.14`
* `gym`
* `stable_baselines`
* `imageio`
* `matplotlib` (for basic debug mode rendering)

## Usage

Refer to example here(under construction)

Before running the script, cd to the root directory of `code-org` game repos, build with `yarn install` and tell `pm2` to spawn the NodeJS server for the game.

`$ pm2 start ./src/main.js --name "game-env-1" -- -g frog`

After finishing, stop and delete the process.

`$ pm2 stop game-env-1`

`$ pm2 delete game-env-1`

Note: In current version, port number 3300 is hardcoded, and the batch number is fixed to 1. This will soon change.

Note2: We can also automate the "telling `pm2`" in python. This may later be implemented.

All 4 game instances are subclass of `baselines.common.vec_env.VecEnv` with
the following methods

* `reset() => List[obs]`
  * Restart all games in the batch
* `reset(envs_to_reset: np.ndarray or List[int]) => List[obs] `  
  * Reset some games in the batch
* `step_async(actions: np.ndarray or List[int]) => None` 
  * Send batch action to the game server
  * __Undefined behavior__ if `any(env.buf_done) == True`
* `step_wait() => (List[obs], List[rewards], List[dones], List[infos])` 
  * Send batch action to the game server
* `close() => None`
  * Terminate the batch
* `render() => None`
  * View _all_ games in the batch through matplotlib. Available only when `env.pixel == True`.
* `render(i: int) => None`
  * View i-th game in the batch through matplotlib. Available only when `env.pixel == True`.

### Apple and Color

`AppleVecEnv(num_envs=1, js_filename="apple1", pixel=True,
                 mouse_action=True, scale_action=False)`

####Params

* `mouse_action` (bool): Set to `False` if the action is picking one of the 20 tiles in the picture ([0-19]). Set to `True` if action is a mouse click at position `(x,y)`. 
* `scale_action`(bool): Set to `True` if action is to be scaled to range $[0,1]^2$ (relative position. [1,1] in scaled mode is equivalent to [max_w-1, max_h-1] in unscaled mode). Can be done only if `mouse_action == True`

#### Step actions

* `List[int]` containing integers in range [0-19] inclusive when `mouse_action==True`
* `List[[float,float]]` containing the x-y coordinate(s) to click on the screen(s).
  * Range is [0,1] if  `scale_action==True`
  * Range is [0, max_w-1] and [0, max_h-1]  if  `scale_action==False`

#### Step returns

##### Observations

* For `pixel==True`: `np.ndarray(shape=(num_envs, max_H, max_W, 3), dtype=np.uint8)` The RGB array representing the JPEG screenshot of the game.
* For `pixel==False`: `np.ndarray(shape=(num_envs, num_objs, 2), dtype=np.float32)` The x-y positions of all objects on the screen, sorted by `id` (name) alphabetically.

##### Rewards

* `np.ndarray(shape=(num_env,), dtype=np.float32)` Stepwise increment in score, extracted from the json response.

##### Dones

* `np.ndarray(shape=(num_env,), dtype=np.bool)`. Representing whether the game has ended. I.e. when the player is damaged one more time _after_ lives = 0.

### Color

`ColorVecEnv(num_envs=1, js_filename="color1", pixel=True,
                 mouse_action=True, scale_action=False)`

#### Params

* `mouse_action` (bool): Set to `False` if the action is picking one of the 20 tiles in the picture ([0-19]). Set to `True` if action is a mouse click at position `(x,y)`. 
* `scale_action`(bool): Set to `True` if action is to be scaled to range $[0,1]^2$ (relative position. [1,1] in scaled mode is equivalent to [max_w-1, max_h-1] in unscaled mode). Can be done only if `mouse_action == True`

#### Step actions

* `List[int]` containing integers in range [0-19] inclusive when `mouse_action==True`
* `List[[float,float]]` containing the x-y coordinate(s) to click on the screen(s).
  * Range is [0,1] if  `scale_action==True`
  * Range is [0, max_w-1] and [0, max_h-1]  if  `scale_action==False`

#### Step returns

##### Observations

* For `pixel==True`: `np.ndarray(shape=(num_envs, max_H, max_W, 3), dtype=np.uint8)` The RGB array
* For `pixel==False`: `np.ndarray(shape=(num_envs, num_objs_with_bgcolor, 4), dtype=np.float32)` The RGBA color encoding of all objects _with attribute_ `backgroundColor` on the screen, sorted by `id` (name) alphabetically.

##### Rewards

* `np.ndarray(shape=(num_env,), dtype=np.float32)` Stepwise _increment_ in score for the _most recent_ player, extracted from the json response.

##### Dones

* `np.ndarray(shape=(num_env,), dtype=np.bool)`. Representing whether the game has ended. Always False

### Frog

`FrogVecEnv(num_envs=1, js_filename="frog1", pixel=True, frames=2)`

#### Params

* `frames` (int) defining number of frames in the game for one action step. Default is 2.

#### Step actions

* `List[int]` containing integers in range [0-15] inclusive

  * +1 to hold LEFT key
  * +2 to hold UP key
  * +4 to hold RIGHT key
  * +8 to hold DOWN key

  For example, 0 means no key, 3 means LEFT and UP

#### Step returns

##### Observations

* For `pixel==True`: `np.ndarray(shape=(num_envs, max_H, max_W, 3), dtype=np.uint8)` The RGB array
* For `pixel==False`: `np.ndarray(shape=(num_envs, num_objs, 2), dtype=np.float32)` The x-y positions of all _animated_ objects on the screen, sorted by `id` (name) alphabetically.

##### Rewards

* `np.ndarray(shape=(num_env,), dtype=np.float32)` Stepwise _increment_ in score since last step, extracted from the json response.

##### Dones

* `np.ndarray(shape=(num_env,), dtype=np.bool)`. Representing whether the game has ended. I.e. when the player is damaged one more time _after_ health = 0 and the screen displays "game over" message.

### Flyer

`FlyerVecEnv(num_envs=1, js_filename="flyer1", pixel=True, frames=2)`

#### Params

* `frames` (int) defining number of frames in the game for one action step. Default is 2. Cannot be set to 1 in `Flyer` because of some complications in the tick function implemented in js.

#### Step actions

* `List[int]` containing integers in range [0-15] inclusive

  * +1 to hold LEFT key
  * +2 to hold UP key
  * +4 to hold RIGHT key
  * +8 to hold DOWN key

  For example, 0 means no key is pressed. 2 means UP is pressed. 3 means LEFT and UP are pressed. 15 means all keys at once.

#### Step returns

##### Observations

* For `pixel==True`: `np.ndarray(shape=(num_envs, max_H, max_W, 3), dtype=np.uint8)` The RGB array
* For `pixel==False`: `np.ndarray(shape=(num_envs, num_objs, 2), dtype=np.float32)` The x-y positions of all _animated_ objects on the screen, sorted by `id` (name) alphabetically.

##### Rewards

* `np.ndarray(shape=(num_env,), dtype=np.float32)` Extracted from whether the coin changed position compared to the last step.

##### Dones

* `np.ndarray(shape=(num_env,), dtype=np.bool)`. Representing whether the game has ended. I.e. when the player went off screen and the screen displays "game over" message.