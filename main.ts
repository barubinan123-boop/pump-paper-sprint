// Pump.fun Paper Sprint — "Sprint Momentum" strategy.
//
// A paper-trading bot for the agent-fun/replay/v2 protocol. It watches every
// pump.fun bonding curve in the replay window and takes only one shape: a burst
// on a brand-new coin. Fixed take-profits lose to the bleed plus fees in every
// combination tried on the dev tape, so exits are a tight stop-loss plus a wide
// trailing stop with no fixed take-profit: losers cost 3%, and the rare runner
// rides until it gives back 25% from its peak. Position sizes stay small and
// fixed so one bad coin can never sink the run, and the freed slot is
// immediately recycled into the next candidate, which is what lets the bot
// cover many coins in 6 hours.
//
// Rules of the game this bot plays by:
// - Paper trading only: it only ever emits buy/sell orders to the harness.
// - Fully deterministic: every decision comes from the batch data (slots,
//   events, marks, account). No randomness, no wall clock, no network, so two
//   runs of the same window produce the same action log byte for byte.
// - It never chases a coin that already ran away: entries require fresh
//   momentum but cap how far the price may already have risen since first seen.

import { runBot } from './vendor/makefun-bot.ts';
import type { Batch, BotState, OrderInput } from './vendor/makefun-bot.ts';

// ------------------------------------------------------------------ tuning
//
// Tuned against the dev window's tape (offline analysis of the kit dataset):
// pump.fun coins bleed on average at every horizon, so fixed take-profits
// lose to the bleed plus ~2.5% round-trip fees in every combination tried.
// What works is riding the rare runners: about 14% of early-burst signals
// reach +100% within 1200 slots, while losers are cut at -3%. So entries
// chase the early burst and exits are a tight stop-loss plus a wide trailing
// stop with no fixed take-profit — losers cost 3%, winners ride until they
// give back 25% from their peak. Simulated on the dev tape that profile nets
// about +4% per round trip after fees.

/** SOL per entry, in lamports (0.05 SOL). Small and fixed: one rug hurts little,
 * and a small order moves a young curve's price far less than a big one. */
const BUY_LAMPORTS = 50_000_000n;
/** Smallest buy worth paying fees on (0.02 SOL). */
const MIN_BUY_LAMPORTS = 20_000_000n;
/** Trades a coin must show before it can be considered. */
const MIN_TRADES = 4;
/** Trades kept per coin for the momentum / volume read. */
const RING = 12;
/** Entry: share of recent SOL volume that must be buys (60%). */
const ENTRY_BUY_SHARE = 0.6;
/** Entry: the coin's latest trade must be this recent (3600 slots, ~24 min). */
const MAX_TRADE_AGE_SLOTS = 3600;
/** Entry orders tolerate this much slippage (1000 bps): in a fast burst the
 * price runs on within 2 slots of latency, and a tight cap rejects exactly
 * the buys whose run is still going — adverse selection. */
const SLIPPAGE_BPS = 1000;
/** Exit orders tolerate this much slippage (1000 bps, exits must land even in a dump). */
const EXIT_SLIPPAGE_BPS = 1000;

/** Early burst: a coin in its first trades, still pumping on buy tape. */
const EARLY_MAX_TRADES = 30;
const EARLY_MAX_AGE_SLOTS = 1200;
const EARLY_MOMENTUM = 0.05;
const EARLY_MAX_RUN_UP = 0.6;
/** Exit: no fixed take-profit — winners ride under the trailing stop. */
const EARLY_TAKE_PROFIT = 10;
/** Exit: stop loss at -3% under entry. */
const EARLY_STOP_LOSS = 0.03;
/** Exit: trailing stop, 25% below the best price seen since entry. */
const EARLY_TRAILING_DROP = 0.25;
/** Exit: give up on a flat position after this many slots (~8 min). */
const EARLY_MAX_HOLD_SLOTS = 1200;

// ------------------------------------------------------------------- state

interface CoinTrack {
  /** Prices (SOL per token, as a float ratio) of the last RING trades, oldest first. */
  prices: number[];
  /** SOL volume of buys / sells over those trades, in lamports. */
  buyVol: bigint;
  sellVol: bigint;
  /** Price at first sight, for the run-up guard. */
  firstPrice: number;
  /** Slot this coin was first seen. */
  firstSlot: number;
  /** Slot this coin was last seen, for pruning. */
  lastSlot: number;
  /** Slot of its most recent trade, for the recency guard. */
  lastTradeSlot: number;
  /** Trades seen in total. */
  trades: number;
}

interface OpenTrade {
  /** Entry price (SOL per token). */
  entry: number;
  /** Best price seen since entry, for the trailing stop. */
  peak: number;
  /** Slot the buy was decided, for the maximum hold. */
  slot: number;
  /** Exit profile chosen at entry (early burst vs established momentum). */
  takeProfit: number;
  stopLoss: number;
  trailingDrop: number;
  maxHoldSlots: number;
}

/** Price of a curve right now: virtual SOL reserves per virtual token reserve. */
function curvePrice(virtualQuote: bigint, virtualToken: bigint): number {
  if (virtualToken <= 0n || virtualQuote <= 0n) return 0;
  return Number(virtualQuote) / Number(virtualToken);
}

const tracks = new Map<string, CoinTrack>();
const open = new Map<string, OpenTrade>();

function pruneStale(nowSlot: number): void {
  if (tracks.size <= 4000) return;
  for (const [mint, t] of tracks) {
    if (nowSlot - t.lastSlot > 100_000 && !open.has(mint)) tracks.delete(mint);
  }
}

function observeTrade(mint: string, price: number, isBuy: boolean, solAmount: bigint, slot: number): void {
  let t = tracks.get(mint);
  if (t === undefined) {
    t = { prices: [], buyVol: 0n, sellVol: 0n, firstPrice: price, firstSlot: slot, lastSlot: slot, lastTradeSlot: slot, trades: 0 };
    tracks.set(mint, t);
  }
  t.prices.push(price);
  if (t.prices.length > RING) t.prices.shift();
  if (isBuy) t.buyVol += solAmount;
  else t.sellVol += solAmount;
  t.trades += 1;
  t.lastSlot = slot;
  t.lastTradeSlot = slot;
}

function observeMark(mint: string, price: number, slot: number): void {
  // A mark only notes the coin is still alive; the ring keeps trade prices so
  // the momentum read never comes from a mark snapshot.
  const t = tracks.get(mint);
  if (t === undefined) return;
  t.lastSlot = slot;
}

/** Fresh momentum: rise from the oldest price in the ring to the latest. */
function ringMomentum(t: CoinTrack): number {
  if (t.prices.length < 2) return 0;
  const first = t.prices[0];
  const last = t.prices[t.prices.length - 1];
  if (first <= 0 || last <= 0) return 0;
  return last / first - 1;
}

/** Is the price still rising over the last few trades (no immediate reversal)? */
function stillRising(t: CoinTrack): boolean {
  if (t.prices.length < 4) return true;
  const last = t.prices[t.prices.length - 1];
  const prev = t.prices[t.prices.length - 4];
  return last >= prev;
}

function buyShare(t: CoinTrack): number {
  const total = t.buyVol + t.sellVol;
  if (total <= 0n) return 0;
  return Number(t.buyVol) / Number(total);
}

// -------------------------------------------------------------------- bot

await runBot({
  onBatch(state: BotState, batch: Batch): OrderInput[] {
    const orders: OrderInput[] = [];

    // Fold this batch's tape into the per-coin tracks.
    for (const e of batch.events) {
      if (e.event === 'TradeEvent') {
        const price = curvePrice(BigInt(e.virtual_sol_reserves), BigInt(e.virtual_token_reserves));
        if (price > 0) observeTrade(e.mint, price, e.is_buy, BigInt(e.sol_amount), e.slot);
      }
    }
    for (const m of batch.marks) {
      const price = curvePrice(BigInt(m.virtual_quote_reserves), BigInt(m.virtual_token_reserves));
      if (price > 0) observeMark(m.mint, price, batch.slot);
    }
    pruneStale(batch.slot);

    // Correct entry prices to what the fills actually paid: a buy resolves
    // 2 slots after the decision, against the slot's last trade and moved by
    // the order's own impact, so the fill runs ~7% above the decision price
    // on average. Measuring the stop-loss and trailing stop from the decision
    // price instead lets losers run ~7% deeper than intended before exiting.
    for (const f of batch.fills) {
      if (f.side === 'buy' && f.status === 'filled') {
        const tokens = BigInt(f.token_amount);
        const quote = BigInt(f.quote_lamports);
        if (tokens > 0n && quote > 0n) {
          const fillPrice = Number(quote) / Number(tokens);
          const trade = open.get(f.mint);
          if (trade !== undefined) {
            trade.entry = fillPrice;
            if (fillPrice > trade.peak) trade.peak = fillPrice;
          }
        }
      }
      if (f.side === 'sell' && f.status === 'filled') open.delete(f.mint);
    }

    // Drop entry records for buys that never became a position (rejected or
    // failed fills): without this a dead buy would block its mint forever.
    for (const mint of [...open.keys()]) {
      if (state.positions.has(mint)) continue;
      let buyPending = false;
      for (const p of state.pending.values()) {
        if (p.order.side === 'buy' && p.order.mint === mint) {
          buyPending = true;
          break;
        }
      }
      if (!buyPending) open.delete(mint);
    }

    // --- exits first: free slots before looking for new entries.
    for (const [mint, holding] of state.positions) {
      const curve = state.curves.get(mint);
      if (curve === undefined) continue;
      const price = curvePrice(curve.virtual_quote_reserves, curve.virtual_token_reserves);
      if (price <= 0) continue;
      const trade = open.get(mint);
      if (trade === undefined) {
        // A position with no entry record (should not happen): track from here.
        open.set(mint, {
          entry: price,
          peak: price,
          slot: batch.slot,
          takeProfit: EARLY_TAKE_PROFIT,
          stopLoss: EARLY_STOP_LOSS,
          trailingDrop: EARLY_TRAILING_DROP,
          maxHoldSlots: EARLY_MAX_HOLD_SLOTS,
        });
        continue;
      }
      if (price > trade.peak) trade.peak = price;
      const gain = price / trade.entry - 1;
      const offPeak = price / trade.peak - 1;
      const held = batch.slot - trade.slot;
      const exit =
        gain >= trade.takeProfit ||
        gain <= -trade.stopLoss ||
        offPeak <= -trade.trailingDrop ||
        held >= trade.maxHoldSlots;
      // A sell already in flight covers this position: do not send another.
      // The entry record is dropped when the sell's fill lands, so a failed
      // or rejected sell is retried on the next batch with its cost intact.
      let sellPending = false;
      for (const p of state.pending.values()) {
        if (p.order.side === 'sell' && p.order.mint === mint) {
          sellPending = true;
          break;
        }
      }
      if (exit && !sellPending) {
        orders.push({
          side: 'sell',
          mint,
          token_amount: holding.token_amount.toString(),
          max_slippage_bps: EXIT_SLIPPAGE_BPS,
        });
      }
    }

    // --- entries: two fast shapes only — an early burst, or sustained
    // momentum on an established coin. Everything else bleeds on average.
    const freeSlots = state.hello.caps.max_open_positions - state.committedPositionCount();
    if (freeSlots > 0) {
      let buys = 0;
      for (const [mint, t] of tracks) {
        if (buys >= freeSlots) break;
        if (orders.length >= state.hello.caps.max_orders_per_batch) break;
        if (state.positions.has(mint) || open.has(mint)) continue;
        let buyPending = false;
        for (const p of state.pending.values()) {
          if (p.order.side === 'buy' && p.order.mint === mint) {
            buyPending = true;
            break;
          }
        }
        if (buyPending) continue;
        const curve = state.curves.get(mint);
        if (curve === undefined || curve.complete) continue;
        if (t.trades < MIN_TRADES) continue;
        if (batch.slot - t.lastTradeSlot > MAX_TRADE_AGE_SLOTS) continue;
        const price = t.prices[t.prices.length - 1];
        if (price <= 0) continue;
        if (buyShare(t) < ENTRY_BUY_SHARE) continue;
        if (!stillRising(t)) continue;

        const age = batch.slot - t.firstSlot;
        const mom = ringMomentum(t);
        // Shape 1: early burst — first trades, still pumping, not run away yet.
        // (The only shape with a positive mean forward return from the fill
        // price in the dev tape, and only inside ~60 slots: exit fast.)
        let profile: Pick<OpenTrade, 'takeProfit' | 'stopLoss' | 'trailingDrop' | 'maxHoldSlots'> | null = null;
        if (
          t.trades <= EARLY_MAX_TRADES &&
          age <= EARLY_MAX_AGE_SLOTS &&
          mom >= EARLY_MOMENTUM &&
          price / t.firstPrice - 1 <= EARLY_MAX_RUN_UP
        ) {
          profile = {
            takeProfit: EARLY_TAKE_PROFIT,
            stopLoss: EARLY_STOP_LOSS,
            trailingDrop: EARLY_TRAILING_DROP,
            maxHoldSlots: EARLY_MAX_HOLD_SLOTS,
          };
        }
        if (profile === null) continue;

        // Size: fixed stake, capped by the harness order-size cap and free cash.
        const cap = state.maxBuyLamports(mint);
        let size = BUY_LAMPORTS;
        if (size > cap) size = cap;
        const free = state.freeCashLamports();
        if (size > free) size = free;
        if (size < MIN_BUY_LAMPORTS) continue;

        orders.push({ side: 'buy', mint, quote_lamports: size.toString(), max_slippage_bps: SLIPPAGE_BPS });
        open.set(mint, { entry: price, peak: price, slot: batch.slot, ...profile });
        buys += 1;
      }
    }

    return orders;
  },
});
