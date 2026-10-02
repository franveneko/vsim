'use strict'
// The Any% route, expressed in real Minecraft item and block names.
// Same 19 milestones as the sandbox, so a brain evolved there can be dropped
// straight into the real game without relearning what the stages mean.

const LOGS = ['oak_log', 'birch_log', 'spruce_log', 'jungle_log', 'acacia_log',
  'dark_oak_log', 'mangrove_log', 'cherry_log']
const PLANKS = LOGS.map(l => l.replace('_log', '_planks'))

const MILESTONES = [
  'punch_wood', 'craft_planks', 'place_bench', 'wooden_pickaxe',
  'mine_stone', 'stone_pickaxe', 'mine_iron', 'smelt_iron',
  'iron_pickaxe', 'mine_diamond', 'diamond_pickaxe', 'mine_obsidian',
  'light_portal', 'enter_nether', 'blaze_rods', 'ender_pearls',
  'craft_eyes', 'enter_end', 'slay_dragon'
]

// Which block each stage sends the bot looking for.
const GOAL_BLOCK = {
  punch_wood: LOGS, craft_planks: LOGS, place_bench: LOGS, wooden_pickaxe: LOGS,
  mine_stone: ['stone', 'cobblestone', 'deepslate'],
  stone_pickaxe: ['stone', 'cobblestone', 'deepslate'],
  mine_iron: ['iron_ore', 'deepslate_iron_ore'],
  smelt_iron: ['iron_ore', 'deepslate_iron_ore'],
  iron_pickaxe: ['iron_ore', 'deepslate_iron_ore'],
  mine_diamond: ['diamond_ore', 'deepslate_diamond_ore'],
  diamond_pickaxe: ['diamond_ore', 'deepslate_diamond_ore'],
  mine_obsidian: ['obsidian', 'lava'],
  light_portal: ['obsidian'],
  enter_nether: ['nether_portal'],
  blaze_rods: ['nether_bricks', 'spawner'],
  ender_pearls: ['nether_bricks'],
  craft_eyes: ['nether_bricks'],
  enter_end: ['end_portal', 'end_portal_frame'],
  slay_dragon: ['end_stone', 'obsidian']
}

const HAZARD = new Set(['lava', 'fire', 'magma_block', 'cactus', 'sweet_berry_bush'])

// The counters the sandbox's observation exposes, mapped onto real item ids.
const ITEM_GROUPS = {
  log: LOGS, planks: PLANKS, stick: ['stick'],
  cobble: ['cobblestone', 'stone', 'cobbled_deepslate'],
  iron_ore: ['raw_iron', 'iron_ore', 'deepslate_iron_ore'],
  iron: ['iron_ingot'], diamond: ['diamond'], obsidian: ['obsidian'],
  blaze_rod: ['blaze_rod'], pearl: ['ender_pearl'], eye: ['ender_eye']
}

const PICKAXES = { wooden_pickaxe: 1, stone_pickaxe: 2, iron_pickaxe: 3, diamond_pickaxe: 4, netherite_pickaxe: 5 }

function count (bot, group) {
  const names = ITEM_GROUPS[group] || []
  let n = 0
  for (const item of bot.inventory.items()) if (names.includes(item.name)) n += item.count
  return n
}

function tier (bot) {
  let t = 0
  for (const item of bot.inventory.items()) t = Math.max(t, PICKAXES[item.name] || 0)
  return t
}

/** Which milestones are satisfied right now, in route order. */
function reached (bot, state) {
  const c = g => count(bot, g)
  const dim = bot.game.dimension
  return [
    c('log') >= 1 || state.everLog,
    c('planks') >= 1 || state.everPlanks,
    state.benchPlaced,
    tier(bot) >= 1,
    c('cobble') >= 1 || state.everCobble,
    tier(bot) >= 2,
    c('iron_ore') >= 1 || state.everIronOre,
    c('iron') >= 1 || state.everIron,
    tier(bot) >= 3,
    c('diamond') >= 1 || state.everDiamond,
    tier(bot) >= 4,
    c('obsidian') >= 10 || state.everObsidian,
    state.portalLit,
    dim.includes('nether') || state.everNether,
    c('blaze_rod') >= 2 || state.everRods,
    c('pearl') >= 2 || state.everPearls,
    c('eye') >= 2 || state.everEyes,
    dim.includes('end') || state.everEnd,
    state.dragonDown
  ]
}

/** Stage index: the first milestone not yet done, gated in order. */
function stage (done) {
  let i = 0
  while (i < done.length && done[i]) i++
  return i
}

module.exports = { MILESTONES, GOAL_BLOCK, HAZARD, ITEM_GROUPS, LOGS, PLANKS,
  count, tier, reached, stage }
