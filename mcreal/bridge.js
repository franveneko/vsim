'use strict'
/**
 * A real Minecraft bot that speaks the same 32-number observation and the same
 * eight actions as the MineSim sandbox the flies were evolved in.
 *
 * Nothing about the game is simulated here: the world, the physics, the block
 * breaking times and the mobs are whatever the server says they are.  This file
 * only translates between the server's view and the vector the connectome
 * expects, so a brain trained in the fast sandbox can be dropped into the real
 * game without being retrained.
 *
 * Protocol: newline-delimited JSON over TCP.  Python asks, the bridge answers.
 *   {"cmd":"reset"}            -> {"obs":[...], "info":{...}}
 *   {"cmd":"step","a":0..7}    -> {"obs":[...], "info":{...}}
 *   {"cmd":"quit"}             -> closes
 */
const net = require('net')
const mineflayer = require('mineflayer')
const { Vec3 } = require('vec3')
const R = require('./route.js')

const OBS_DIM = 32
const ACTIONS = ['north', 'south', 'east', 'west', 'mine', 'place', 'craft', 'attack']
// MineSim's four directions, in Minecraft's axes: its y axis is Minecraft's z.
const DIRS = [{ x: 0, z: -1 }, { x: 0, z: 1 }, { x: 1, z: 0 }, { x: -1, z: 0 }]

const opts = Object.fromEntries(process.argv.slice(2).map(s => s.split('=')))
const HOST = opts.host || '127.0.0.1'
const PORT = parseInt(opts.port || '25565', 10)
const VERSION = opts.version || '1.18.2'
const USERNAME = opts.username || 'fly0'
const BRIDGE_PORT = parseInt(opts.bridgePort || '8765', 10)
const MAX_TICKS = parseInt(opts.maxTicks || '4000', 10)
// 'offline' for a LAN/offline-mode server; 'microsoft' signs in with a real
// Minecraft account, which is what an online-mode server requires.
const AUTH = opts.auth || 'offline'

const clip = (v, lo, hi) => Math.max(lo, Math.min(hi, v))
const sleep = ms => new Promise(r => setTimeout(r, ms))

class FlyBot {
  constructor (bot) {
    this.bot = bot
    this.facing = 0
    this.tick = 0
    this.mineProgress = 0
    this.prevMobDist = 99
    this.state = { benchPlaced: false, portalLit: false, dragonDown: false }
    this.lastAction = -1
    this.log = []
  }

  // ------------------------------------------------------------- observation
  blockAt (dx, dy, dz) {
    const p = this.bot.entity.position
    return this.bot.blockAt(new Vec3(Math.floor(p.x) + dx, Math.floor(p.y) + dy,
      Math.floor(p.z) + dz))
  }

  goalNames () {
    const names = R.GOAL_BLOCK[R.MILESTONES[this.stage()]] || []
    return Array.isArray(names) ? names : [names]
  }

  goalBlock () {
    const names = this.goalNames()
    return this.bot.findBlock({
      matching: b => b && names.includes(b.name), maxDistance: 48, count: 1
    })
  }

  /** Milestones latch: picking up a log and then spending it on planks must not
   *  un-reach `punch_wood`.  The sandbox gates stages the same way. */
  updateLatches () {
    const c = g => R.count(this.bot, g)
    const st = this.state
    if (c('log') >= 1) st.everLog = true
    if (c('planks') >= 1) st.everPlanks = true
    if (c('cobble') >= 1) st.everCobble = true
    if (c('iron_ore') >= 1) st.everIronOre = true
    if (c('iron') >= 1) st.everIron = true
    if (c('diamond') >= 1) st.everDiamond = true
    if (c('obsidian') >= 10) st.everObsidian = true
    if (c('blaze_rod') >= 2) st.everRods = true
    if (c('pearl') >= 2) st.everPearls = true
    if (c('eye') >= 2) st.everEyes = true
    const dim = this.bot.game.dimension || ''
    if (dim.includes('nether')) st.everNether = true
    if (dim.includes('end')) st.everEnd = true
  }

  done () {
    this.updateLatches()
    return R.reached(this.bot, this.state)
  }

  stage () {
    return R.stage(this.done())
  }

  nearestMob () {
    const p = this.bot.entity.position
    let best = null, bestD = 1e9
    for (const id of Object.keys(this.bot.entities)) {
      const e = this.bot.entities[id]
      if (!e || e === this.bot.entity) continue
      if (e.type !== 'mob' && e.type !== 'hostile' && e.type !== 'player') continue
      const d = e.position.distanceTo(p)
      if (d < bestD) { bestD = d; best = e }
    }
    return { entity: best, dist: best ? bestD : 999 }
  }

  observe () {
    const bot = this.bot
    const o = new Array(OBS_DIM).fill(0)
    const goalNames = this.goalNames()

    for (let d = 0; d < 4; d++) {
      const b = this.blockAt(DIRS[d].x, 0, DIRS[d].z)
      o[d * 3 + 0] = b && b.boundingBox === 'block' ? 1 : 0
      o[d * 3 + 1] = b && R.HAZARD.has(b.name) ? 1 : 0
      o[d * 3 + 2] = b && goalNames.includes(b.name) ? 1 : 0
    }

    const p = bot.entity.position
    const g = this.goalBlock()
    const gd = g ? Math.abs(g.position.x - p.x) + Math.abs(g.position.z - p.z) : 40
    o[12] = clip(g ? (g.position.x - p.x) / 8 : 0, -1, 1)
    o[13] = clip(g ? (g.position.z - p.z) / 8 : 0, -1, 1)
    o[14] = clip(1 - gd / 32, 0, 1)

    const m = this.nearestMob()
    const near = m.dist < 40
    o[15] = near ? clip((m.entity.position.x - p.x) / 8, -1, 1) : 0
    o[16] = near ? clip((m.entity.position.z - p.z) / 8, -1, 1) : 0
    o[17] = near ? clip(1 - m.dist / 12, 0, 1) : 0
    o[18] = clip(this.prevMobDist - m.dist, 0, 3) / 3
    this.prevMobDist = m.dist

    o[19] = R.tier(bot) / 4
    o[20] = this.stage() / R.MILESTONES.length
    const groups = ['log', 'cobble', 'iron', 'diamond']
    for (let i = 0; i < 4; i++) o[21 + i] = clip(R.count(bot, groups[i]) / 4, 0, 1)
    o[25] = (bot.health || 0) / 20
    const dim = bot.game.dimension || 'overworld'
    o[26] = dim.includes('nether') || dim.includes('end') ? 0 : 1
    o[27] = dim.includes('nether') ? 1 : 0
    o[28] = dim.includes('end') ? 1 : 0
    o[29] = clip(this.mineProgress / 10, 0, 1)
    o[30] = clip(this.tick / MAX_TICKS, 0, 1)
    o[31] = 1
    return o
  }

  bearing () {
    const g = this.goalBlock()
    if (!g) return 0
    const p = this.bot.entity.position
    return Math.atan2(g.position.z - p.z, g.position.x - p.x)
  }

  // ----------------------------------------------------------------- actions
  async move (d) {
    const bot = this.bot
    this.facing = d
    const yaw = Math.atan2(-DIRS[d].x, -DIRS[d].z)
    await bot.look(yaw, 0, true)
    const before = bot.entity.position.clone()
    bot.setControlState('forward', true)
    await sleep(220)
    // a blocked step means something is in the way: hop it
    if (bot.entity.position.distanceTo(before) < 0.15) {
      bot.setControlState('jump', true)
      await sleep(180)
      bot.setControlState('jump', false)
    }
    bot.setControlState('forward', false)
    this.mineProgress = 0
    const moved = bot.entity.position.distanceTo(before)
    return `moved ${moved.toFixed(2)} ${ACTIONS[d]}`
  }

  /** The block this swing lands on.
   *
   *  In the sandbox the brain learned in, the ground is floor you walk on and
   *  cannot be mined; only blocks that stand in your way or that the route
   *  wants are targets.  Keeping that true here is what makes the learned
   *  behaviour transfer: without it the bot spends the run digging a hole in
   *  the grass, because in its world that was never an option.
   *
   *  So: whatever the milestone wants and is in reach, else the block actually
   *  blocking the step we face.  Never the floor, never an arbitrary neighbour.
   */
  digTarget () {
    const goalNames = this.goalNames()
    const breakable = b => b && b.boundingBox === 'block' && b.name !== 'bedrock' &&
      this.bot.canDigBlock(b)

    const reach = []
    for (const dy of [0, 1, -1]) {
      for (const d of DIRS) reach.push(this.blockAt(d.x, dy, d.z))
      reach.push(this.blockAt(0, dy === 0 ? 2 : dy, 0))
    }
    const wanted = reach.find(b => breakable(b) && goalNames.includes(b.name))
    if (wanted) return wanted

    const faced = this.blockAt(DIRS[this.facing].x, 0, DIRS[this.facing].z)
    return breakable(faced) ? faced : null
  }

  async mine () {
    const target = this.digTarget()
    if (!target) return 'nothing to dig'
    try {
      await this.bot.dig(target, true)
      this.mineProgress = 0
      return `dug ${target.name}`
    } catch (e) {
      this.mineProgress = Math.min(10, this.mineProgress + 1)
      return `dig failed: ${e.message}`
    }
  }

  async place () {
    const bot = this.bot
    const st = R.MILESTONES[this.stage()]
    const want = { place_bench: 'crafting_table', light_portal: 'obsidian',
      enter_end: 'ender_eye' }[st]
    if (!want) return 'nothing to place at this stage'
    const item = bot.inventory.items().find(i => i.name === want)
    if (!item) return `no ${want}`
    const ref = this.blockAt(0, -1, 0)
    if (!ref || ref.boundingBox !== 'block') return 'no surface to place on'
    try {
      await bot.equip(item, 'hand')
      await bot.placeBlock(ref, new Vec3(0, 1, 0))
      if (want === 'crafting_table') this.state.benchPlaced = true
      return `placed ${want}`
    } catch (e) { return `place failed: ${e.message}` }
  }

  async craft () {
    const bot = this.bot
    const st = R.MILESTONES[this.stage()]
    const plan = {
      craft_planks: R.PLANKS, place_bench: ['crafting_table'],
      wooden_pickaxe: ['stick', 'crafting_table', 'wooden_pickaxe'],
      stone_pickaxe: ['stick', 'stone_pickaxe'],
      smelt_iron: ['iron_ingot'], iron_pickaxe: ['stick', 'iron_pickaxe'],
      diamond_pickaxe: ['stick', 'diamond_pickaxe'], craft_eyes: ['ender_eye']
    }[st] || R.PLANKS
    const table = bot.findBlock({ matching: b => b && b.name === 'crafting_table', maxDistance: 4 })
    for (const name of plan) {
      const item = bot.registry.itemsByName[name]
      if (!item) continue
      const recipes = bot.recipesFor(item.id, null, 1, table || null)
      if (!recipes.length) continue
      try {
        await bot.craft(recipes[0], 1, table || null)
        return `crafted ${name}`
      } catch (e) { return `craft ${name} failed: ${e.message}` }
    }
    return 'no recipe available'
  }

  async attack () {
    const m = this.nearestMob()
    if (!m.entity || m.dist > 4) return 'nothing in reach'
    try {
      await this.bot.lookAt(m.entity.position.offset(0, 1, 0), true)
      this.bot.attack(m.entity)
      if (m.entity.name && m.entity.name.includes('dragon')) {
        if (m.entity.health !== undefined && m.entity.health <= 0) this.state.dragonDown = true
      }
      return `hit ${m.entity.name || m.entity.displayName || 'entity'}`
    } catch (e) { return `attack failed: ${e.message}` }
  }

  async act (a) {
    this.lastAction = a
    this.tick++
    if (a < 4) return this.move(a)
    if (a === 4) return this.mine()
    if (a === 5) return this.place()
    if (a === 6) return this.craft()
    return this.attack()
  }

  /** The JavaScript server generates no trees, so a test world needs them
   *  placed by hand.  This is a fixture, not worldgen - say so when you use it.
   *  A vanilla server needs none of this. */
  async seedWorld (trees = 10, radius = 14) {
    const p = this.bot.entity.position
    const cmds = []
    for (let i = 0; i < trees; i++) {
      const a = (2 * Math.PI * i) / trees
      const x = Math.round(p.x + Math.cos(a) * (6 + (i % 3) * 4))
      const z = Math.round(p.z + Math.sin(a) * (6 + (i % 3) * 4))
      for (let dy = 0; dy < 4; dy++) {
        cmds.push(`/setblock ${x} ${Math.round(p.y) + dy} ${z} oak_log`)
      }
    }
    for (const c of cmds) { this.bot.chat(c); await sleep(25) }
    await sleep(600)
    return `planted ${trees} oak trunks within ${radius} blocks`
  }

  info (note) {
    const p = this.bot.entity.position
    const done = this.done()
    return {
      tick: this.tick,
      stage: this.stage(),
      milestone: R.MILESTONES[Math.min(this.stage(), R.MILESTONES.length - 1)],
      milestones_done: done.filter(Boolean).length,
      pos: [Math.round(p.x), Math.round(p.y), Math.round(p.z)],
      dimension: this.bot.game.dimension,
      health: this.bot.health, food: this.bot.food,
      tier: R.tier(this.bot),
      inventory: this.bot.inventory.items().map(i => `${i.name}x${i.count}`),
      action: this.lastAction >= 0 ? ACTIONS[this.lastAction] : null,
      note: note || null
    }
  }
}

// ------------------------------------------------------------------- wiring
async function main () {
  const bot = mineflayer.createBot({
    host: HOST, port: PORT, username: USERNAME, version: VERSION, auth: AUTH
  })
  bot.on('error', e => console.error('[bot] error', e.message))
  bot.on('kicked', r => console.error('[bot] kicked', JSON.stringify(r)))

  await new Promise(res => bot.once('spawn', res))
  await bot.waitForChunksToLoad()
  const fly = new FlyBot(bot)
  console.error(`[bridge] ${USERNAME} spawned at ${bot.entity.position}`)

  const server = net.createServer(sock => {
    let buf = ''
    sock.on('data', async chunk => {
      buf += chunk
      let i
      while ((i = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 1)
        if (!line.trim()) continue
        let msg
        try { msg = JSON.parse(line) } catch { continue }
        if (msg.cmd === 'quit') { sock.end(); process.exit(0) }
        let note = null
        if (msg.cmd === 'step') note = await fly.act(msg.a | 0)
        if (msg.cmd === 'seedworld') note = await fly.seedWorld(msg.trees || 10)
        sock.write(JSON.stringify({
          obs: fly.observe(), bearing: fly.bearing(), info: fly.info(note)
        }) + '\n')
      }
    })
  })
  server.listen(BRIDGE_PORT, '127.0.0.1',
    () => console.error(`[bridge] listening on ${BRIDGE_PORT}`))
}

main().catch(e => { console.error('[bridge] fatal', e); process.exit(1) })
