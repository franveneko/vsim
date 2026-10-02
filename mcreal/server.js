'use strict'
/** Launch a Minecraft server the bots can join.
 *
 * This is flying-squid, a Minecraft server implemented in JavaScript.  It
 * speaks the genuine Minecraft protocol and generates a genuine world, so a
 * vanilla client can join it and the bots see real blocks and real physics -
 * but it is not Mojang's server, and it does not implement block drops,
 * crafting, mob AI or the End.  Point `--host`/`--port` at a vanilla server
 * instead and everything in the route becomes available.
 */
const squid = require('flying-squid')
const opts = Object.fromEntries(process.argv.slice(2).map(s => s.split('=')))

const server = squid.createMCServer({
  motd: 'vsim - fly connectome', port: parseInt(opts.port || '25565', 10),
  'max-players': parseInt(opts.players || '20', 10), 'online-mode': false,
  logging: true, gameMode: 0, difficulty: 1,
  worldFolder: opts.world || 'world',
  generation: { name: 'diamond_square', options: { worldHeight: 80, seed: parseInt(opts.seed || '42', 10) } },
  kickTimeout: 120000, plugins: {}, modpe: false,
  'view-distance': parseInt(opts.view || '4', 10),
  'player-list-text': { header: 'vsim', footer: 'fly connectome' },
  'everybody-op': true, 'max-entities': 200,
  version: opts.version || '1.18.2'
})
server.on('listening', () => console.log('[server] ready'))
