const squid = require('flying-squid')
const mineflayer = require('mineflayer')
const VERSION = process.argv[2] || '1.18.2'

const server = squid.createMCServer({
  motd: 'vsim', port: 25565, 'max-players': 20, 'online-mode': false,
  logging: false, gameMode: 0, difficulty: 1, worldFolder: 'world',
  generation: { name: 'diamond_square', options: { worldHeight: 80 } },
  kickTimeout: 30000, plugins: {}, modpe: false, 'view-distance': 4,
  'player-list-text': { header: '', footer: '' }, 'everybody-op': true,
  'max-entities': 100, version: VERSION
})

server.on('listening', () => console.log('[server] listening, version', VERSION))

setTimeout(() => {
  const bot = mineflayer.createBot({
    host: '127.0.0.1', port: 25565, username: 'fly0', version: VERSION, auth: 'offline'
  })
  bot.on('error', e => { console.log('[bot] error', e.message); process.exit(1) })
  bot.on('kicked', r => console.log('[bot] kicked', r))
  bot.once('spawn', async () => {
    console.log('[bot] SPAWNED at', bot.entity.position.toString())
    console.log('[bot] game mode', bot.game.gameMode, 'dimension', bot.game.dimension)
    await bot.waitForChunksToLoad()
    const below = bot.blockAt(bot.entity.position.offset(0, -1, 0))
    console.log('[bot] block below:', below && below.name)
    // look around: count distinct block types in a 16-block radius
    const counts = {}
    for (let dx = -8; dx <= 8; dx += 2)
      for (let dy = -6; dy <= 4; dy += 2)
        for (let dz = -8; dz <= 8; dz += 2) {
          const b = bot.blockAt(bot.entity.position.offset(dx, dy, dz))
          if (b) counts[b.name] = (counts[b.name] || 0) + 1
        }
    console.log('[bot] blocks nearby:', JSON.stringify(Object.entries(counts).sort((a,b)=>b[1]-a[1]).slice(0,8)))
    // try digging
    const target = bot.findBlock({ matching: b => b && b.name !== 'air', maxDistance: 4 })
    if (target) {
      console.log('[bot] digging', target.name, 'at', target.position.toString())
      try { await bot.dig(target); console.log('[bot] DIG OK; inventory:', bot.inventory.items().map(i=>i.name+'x'+i.count)) }
      catch (e) { console.log('[bot] dig failed:', e.message) }
    }
    const p0 = bot.entity.position.clone()
    bot.setControlState('forward', true)
    setTimeout(() => {
      bot.setControlState('forward', false)
      console.log('[bot] moved', p0.distanceTo(bot.entity.position).toFixed(2), 'blocks')
      console.log('[bot] health', bot.health, 'food', bot.food)
      console.log('RESULT: OK')
      process.exit(0)
    }, 2500)
  })
}, 4000)

setTimeout(() => { console.log('RESULT: TIMEOUT'); process.exit(2) }, 60000)
