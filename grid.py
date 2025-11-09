"""
Proper Grid Trading Bot for Lighter Exchange
BUY LOW → SELL HIGH → Repeat
"""
import asyncio
import os
import signal
from datetime import datetime
from dotenv import load_dotenv
import lighter
import requests

load_dotenv()

class GridTradingBot:
    # Market symbols
    MARKETS = {
        0: "ETH", 1: "BTC", 2: "SOL", 3: "DOGE", 4: "1000PEPE",
        5: "WIF", 6: "WLD", 7: "XRP", 8: "LINK", 9: "AVAX"
    }

    def __init__(self):
        # Validate required environment variables
        required_vars = ['API_KEY_PRIVATE_KEY', 'ACCOUNT_INDEX', 'API_KEY_INDEX', 'BASE_URL']
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            raise ValueError(f"❌ Missing required environment variables: {', '.join(missing_vars)}\n"
                           f"Please check your .env file!")

        self.api_key_pk = os.getenv('API_KEY_PRIVATE_KEY')

        # Parse and validate numeric values
        try:
            self.account_index = int(os.getenv('ACCOUNT_INDEX'))
            self.api_key_index = int(os.getenv('API_KEY_INDEX'))
            self.market_index = int(os.getenv('MARKET_INDEX', 1))
            self.leverage = int(os.getenv('LEVERAGE', 10))
            self.grid_count = int(os.getenv('GRID_COUNT', 20))
            self.investment = float(os.getenv('INVESTMENT_USDC', 100))
            self.grid_spacing_percent = float(os.getenv('GRID_SPACING_PERCENT', 1.0)) / 100
        except ValueError as e:
            raise ValueError(f"❌ Invalid numeric value in .env file: {e}\n"
                           f"Please check ACCOUNT_INDEX, API_KEY_INDEX, MARKET_INDEX, etc.")

        self.base_url = os.getenv('BASE_URL')
        self.direction = os.getenv('DIRECTION', 'NEUTRAL').upper()
        self.client = None
        self.order_index = 30000
        self.market_symbol = self.MARKETS.get(self.market_index, f"Market{self.market_index}")

        # Grid tracking
        self.running = True
        self.grid_orders = {}  # {client_order_index: {'price': float, 'is_ask': bool, 'base_amount': int}}

        # Statistics
        self.total_profit = 0.0
        self.trades_count = 0
        self.total_volume = 0.0
        self.completed_cycles = 0

        # Market decimals
        self.price_decimals = 4
        self.size_decimals = 8

    async def init(self):
        self.client = lighter.SignerClient(
            url=self.base_url,
            private_key=self.api_key_pk,
            account_index=self.account_index,
            api_key_index=self.api_key_index
        )

        err = self.client.check_client()
        if err:
            raise Exception(f"Client error: {err}")

        print(f"✅ Connected to Lighter")
        print(f"   Account: {self.account_index}")

        # Use timestamp-based order index
        try:
            import time
            self.order_index = int(time.time() * 1000) % 1000000
            print(f"   Starting order index: {self.order_index}")
        except Exception as e:
            print(f"   ⚠️  Could not set timestamp-based order index: {e}")
            # Keep default order_index = 30000

    async def get_market_decimals(self):
        """Get price and size decimals for the current market from API"""
        try:
            url = f"{self.base_url}/api/v1/orderBookOrders?market_id={self.market_index}&limit=1"
            response = requests.get(url)
            data = response.json()

            if data.get('asks') and len(data['asks']) > 0:
                price_str = data['asks'][0]['price']
                if '.' in price_str:
                    self.price_decimals = len(price_str.split('.')[1])
                else:
                    self.price_decimals = 0

                size_str = data['asks'][0].get('remaining_base_amount', '0')
                if '.' in size_str:
                    self.size_decimals = len(size_str.split('.')[1])
                else:
                    self.size_decimals = 8

                print(f"   Market decimals: price={self.price_decimals}, size={self.size_decimals}")
                return True
            else:
                print(f"   ⚠️  Could not fetch market decimals, using defaults")
                return False

        except Exception as e:
            print(f"   ⚠️  Error fetching market decimals: {e}")
            return False

    async def get_current_price(self):
        """Get current price from order book with error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}/api/v1/orderBookOrders?market_id={self.market_index}&limit=1"
                response = requests.get(url, timeout=10)

                if response.status_code != 200:
                    raise Exception(f"API returned status {response.status_code}")

                data = response.json()

                # Validate orderbook data
                if not data.get('bids') or not data.get('asks'):
                    raise Exception("Empty orderbook")

                if len(data['bids']) == 0 or len(data['asks']) == 0:
                    raise Exception("No bids or asks in orderbook")

                best_bid = data['bids'][0]['price']
                best_ask = data['asks'][0]['price']
                current_price = (float(best_bid) + float(best_ask)) / 2

                return current_price, best_bid, best_ask

            except requests.exceptions.Timeout:
                print(f"   ⚠️  Price fetch timeout (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
            except Exception as e:
                print(f"   ⚠️  Price fetch error (attempt {attempt+1}/{max_retries}): {str(e)[:60]}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)

        raise Exception("Failed to fetch current price after 3 attempts")

    def price_to_int(self, price_float):
        """Convert price to int format"""
        price_str = f"{price_float:.{self.price_decimals}f}"
        price_int = int(price_str.replace(".", ""))
        return price_int

    async def calculate_grid_levels(self):
        """Calculate grid price levels based on direction"""
        current_price, best_bid, best_ask = await self.get_current_price()

        # NEUTRAL mode: 1% spacing, equal buy/sell
        grids_per_side = self.grid_count // 2
        grid_levels = []

        # Buy grids (below): -1%, -2%, -3%, etc.
        for i in range(1, grids_per_side + 1):
            price = current_price * (1 - self.grid_spacing_percent * i)
            grid_levels.append(price)

        # Sell grids (above): +1%, +2%, +3%, etc.
        for i in range(1, grids_per_side + 1):
            price = current_price * (1 + self.grid_spacing_percent * i)
            grid_levels.append(price)

        grid_levels.sort()
        self.lower_price = grid_levels[0]
        self.upper_price = grid_levels[-1]

        print(f"\n📊 Grid Setup ({self.direction}):")
        print(f"   {self.market_symbol} Price: ${current_price:,.2f}")
        print(f"   Best Bid: ${best_bid}")
        print(f"   Best Ask: ${best_ask}")
        print(f"   Range: ${self.lower_price:,.2f} - ${self.upper_price:,.2f}")
        print(f"   Grids: {self.grid_count}")
        print(f"   Spacing: {self.grid_spacing_percent * 100}% per grid")
        print(f"   💡 Strategy: BUY LOW → SELL HIGH (proper grid trading)")

        return grid_levels, current_price

    async def place_grid_orders(self, grid_levels, current_price):
        """Place initial grid limit orders"""
        coin_per_order = (self.investment / self.grid_count * self.leverage) / current_price
        base_amount = int(coin_per_order * (10 ** self.size_decimals))

        # Validate minimum order size
        if base_amount == 0:
            min_investment = (current_price / self.leverage / self.grid_count) * (10 ** self.size_decimals) * 2
            raise ValueError(
                f"❌ Order size too small! base_amount = {base_amount}\n"
                f"   Investment: ${self.investment}\n"
                f"   Grid count: {self.grid_count}\n"
                f"   Per order: ${self.investment / self.grid_count:.2f}\n"
                f"   Suggested: Either increase INVESTMENT_USDC to ${min_investment:.2f}+ or reduce GRID_COUNT to {int(self.investment * self.leverage / current_price * (10 ** self.size_decimals) / 2)}+"
            )

        orders_placed = {'buy': 0, 'sell': 0}

        print(f"\n📝 Placing Initial Grid Orders:")
        print(f"   Amount per order: {coin_per_order:.8f} {self.market_symbol}")
        print(f"   Base amount: {base_amount}\n")

        for i, price in enumerate(grid_levels, 1):
            is_ask = price > current_price
            price_int = self.price_to_int(price)

            try:
                tx, tx_hash, err = await self.client.create_order(
                    market_index=self.market_index,
                    client_order_index=self.order_index,
                    base_amount=base_amount,
                    price=price_int,
                    is_ask=is_ask,
                    order_type=lighter.SignerClient.ORDER_TYPE_LIMIT,
                    time_in_force=lighter.SignerClient.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
                    reduce_only=False,
                    trigger_price=0
                )

                if not err:
                    # Track order
                    self.grid_orders[self.order_index] = {
                        'price': price,
                        'is_ask': is_ask,
                        'base_amount': base_amount,
                        'price_int': price_int
                    }
                    orders_placed['sell' if is_ask else 'buy'] += 1
                    print(f"   ✓ {'SELL' if is_ask else 'BUY'} @ ${price:,.2f} (COI: {self.order_index})")
                else:
                    print(f"   ✗ {'SELL' if is_ask else 'BUY'} @ ${price:,.2f} - {str(err)[:80]}")

                self.order_index += 1
                await asyncio.sleep(1.5)

            except Exception as e:
                print(f"   ✗ Error at ${price:,.2f}: {str(e)[:80]}")

        total_placed = orders_placed['buy'] + orders_placed['sell']
        print(f"\n✅ Initial Grid Complete:")
        print(f"   {orders_placed['buy']} buy orders")
        print(f"   {orders_placed['sell']} sell orders")
        print(f"   Total: {total_placed}/{self.grid_count}")

        # Validate all orders were placed
        if total_placed < self.grid_count:
            print(f"\n   ⚠️  WARNING: Only {total_placed}/{self.grid_count} orders placed!")
            print(f"   Missing {self.grid_count - total_placed} orders from initial setup")
            print(f"   This may cause the grid to have fewer orders than expected")

        return orders_placed

    async def get_active_orders(self):
        """Get active orders from API with proper error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                auth_token, err = self.client.create_auth_token_with_expiry()
                if err:
                    print(f"   ⚠️  Auth error (attempt {attempt+1}/{max_retries}): {err}")
                    await asyncio.sleep(2)
                    continue

                url = f"{self.base_url}/api/v1/accountActiveOrders?account_index={self.account_index}&market_id={self.market_index}"
                headers = {"Authorization": auth_token}
                response = requests.get(url, headers=headers, timeout=10)

                if response.status_code != 200:
                    print(f"   ⚠️  API error {response.status_code} (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(2)
                    continue

                data = response.json()

                if 'orders' in data:
                    return data['orders']
                return []

            except requests.exceptions.Timeout:
                print(f"   ⚠️  API timeout (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(2)
            except Exception as e:
                print(f"   ⚠️  API error (attempt {attempt+1}/{max_retries}): {str(e)[:60]}")
                await asyncio.sleep(2)

        # After all retries failed, raise exception to trigger monitor pause
        raise Exception("Failed to fetch active orders after 3 attempts")

    async def monitor_and_refill(self):
        """Monitor orders and refill with PROPER GRID LOGIC"""
        print(f"\n🔄 Proper Grid Trading Started")
        print(f"   Strategy: BUY LOW → SELL HIGH")
        print(f"   Monitoring every 2 seconds")
        print(f"   Press Ctrl+C to stop")
        print(f"   Auto-exit after 10 consecutive errors (with cleanup)")
        print(f"   ⏳ Waiting 5s for orders to sync...\n")

        await asyncio.sleep(5)
        check_interval = 2
        first_check = True
        sync_counter = 0
        consecutive_errors = 0
        max_consecutive_errors = 10
        heartbeat_counter = 0

        while self.running:
            try:
                active_orders = await self.get_active_orders()

                # Reset error counter on success
                consecutive_errors = 0

                # Build set of active client order indices
                active_coi_set = set()
                for order in active_orders:
                    coi = order.get('client_order_index')
                    if coi is not None:  # Only add valid COI, skip None/missing
                        try:
                            coi = int(coi)
                            if coi > 0:  # Skip 0 as it's not a valid COI from our bot
                                active_coi_set.add(coi)
                        except (ValueError, TypeError):
                            pass  # Skip invalid COI

                # Debug on first check
                if first_check:
                    print(f"   📋 Tracking {len(self.grid_orders)} orders")
                    print(f"   📊 Active in API: {len(active_orders)} orders\n")
                    first_check = False

                # Check for filled orders FIRST (before cleanup)
                filled_orders = []
                for coi, order_info in list(self.grid_orders.items()):
                    if coi not in active_coi_set:
                        filled_orders.append((coi, order_info))

                # SAFETY: Limit check to prevent position accumulation
                if len(filled_orders) > 5:
                    print(f"   ⚠️  WARNING: {len(filled_orders)} orders disappeared at once!")
                    print(f"   This might indicate cancellations, not fills. Pausing...")
                    await asyncio.sleep(10)

                # Process filled orders with PROPER GRID LOGIC
                for coi, order_info in filled_orders:
                    filled_price = order_info['price']
                    was_ask = order_info['is_ask']
                    base_amount = order_info['base_amount']

                    # Calculate volume
                    coin_amount = base_amount / (10 ** self.size_decimals)
                    volume_usd = coin_amount * filled_price
                    self.total_volume += volume_usd
                    self.trades_count += 1

                    # SAFETY: Check position before placing refill
                    # Grid bot should maintain near-zero position
                    skip_refill = False
                    try:
                        current_position = await self.get_current_position()
                        max_safe_position = base_amount * 3  # Allow max 3x single order size

                        if abs(current_position) > max_safe_position:
                            print(f"   ⚠️  Position too large: {current_position}")
                            print(f"   Will still attempt refill but monitor position")
                            # Don't skip, just warn - let the refill happen
                    except:
                        pass  # If position check fails, continue anyway

                    refill_success = False
                    if was_ask:
                        # SELL order filled → Place BUY at LOWER price
                        print(f"   💰 SELL @ ${filled_price:,.2f} filled | Volume: ${volume_usd:.0f}")

                        # Calculate new BUY price: grid_spacing BELOW sell price
                        new_buy_price = filled_price * (1 - self.grid_spacing_percent)

                        print(f"      → Placing BUY @ ${new_buy_price:,.2f} (to buy back {self.grid_spacing_percent*100}% cheaper)")

                        refill_success = await self.refill_order(new_buy_price, False, base_amount)

                        # Track profit
                        estimated_profit = (filled_price - new_buy_price) * coin_amount
                        self.total_profit += estimated_profit
                        self.completed_cycles += 0.5

                    else:
                        # BUY order filled → Place SELL at HIGHER price
                        print(f"   📈 BUY @ ${filled_price:,.2f} filled | Volume: ${volume_usd:.0f}")

                        # Calculate new SELL price: grid_spacing ABOVE buy price
                        new_sell_price = filled_price * (1 + self.grid_spacing_percent)

                        print(f"      → Placing SELL @ ${new_sell_price:,.2f} (to sell {self.grid_spacing_percent*100}% higher)")

                        refill_success = await self.refill_order(new_sell_price, True, base_amount)
                        self.completed_cycles += 0.5

                    # Remove filled order from tracking
                    del self.grid_orders[coi]

                    # Alert if refill failed
                    if not refill_success:
                        print(f"      🚨 CRITICAL: Refill FAILED after retries! Grid has {len(self.grid_orders)} orders (expected {self.grid_count})")

                # Periodic sync: Clean up tracking AFTER processing fills (every 15 cycles = 30 seconds)
                sync_counter += 1
                if sync_counter >= 15:
                    # Remove any stale orders that weren't processed as fills
                    # This catches orders cancelled manually on the exchange
                    stale_orders = [coi for coi in self.grid_orders.keys() if coi not in active_coi_set]
                    if stale_orders:
                        for coi in stale_orders:
                            del self.grid_orders[coi]
                        print(f"   🧹 Cleaned {len(stale_orders)} stale tracked orders (manually cancelled)")
                    sync_counter = 0

                # Heartbeat logging every 30 cycles (60 seconds)
                heartbeat_counter += 1
                if heartbeat_counter >= 30:
                    tracked_orders = len(self.grid_orders)
                    order_status = "✅" if tracked_orders >= self.grid_count else "⚠️"
                    print(f"   💚 Bot healthy | {order_status} Orders: {tracked_orders}/{self.grid_count} | Cycles: {self.completed_cycles:.1f} | Profit: ${self.total_profit:.2f}")

                    # Alert if orders are significantly below expected
                    if tracked_orders < self.grid_count - 2:
                        print(f"   🚨 WARNING: Missing {self.grid_count - tracked_orders} orders! Expected {self.grid_count}, have {tracked_orders}")

                    heartbeat_counter = 0

                await asyncio.sleep(check_interval)

            except Exception as e:
                consecutive_errors += 1
                print(f"   ⚠️  Monitor error ({consecutive_errors}/{max_consecutive_errors}): {e}")

                if consecutive_errors >= max_consecutive_errors:
                    print(f"\n   ❌ CRITICAL: {max_consecutive_errors} consecutive errors!")
                    print(f"   🛑 Stopping bot and canceling all orders for safety...")
                    self.running = False
                    break

                await asyncio.sleep(check_interval)

    async def refill_order(self, price, is_ask, base_amount):
        """Place new order at specified price with retry logic"""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                # Check if order already exists at this price level (in bot tracking)
                price_threshold = price * 0.001  # 0.1% tolerance
                for order_info in self.grid_orders.values():
                    if (abs(order_info['price'] - price) < price_threshold and
                        order_info['is_ask'] == is_ask):
                        print(f"      ⚠️  Order already exists at ${price:.2f}, skipping duplicate")
                        return True  # Return success since order exists

                # SAFETY: Check active orders from API (with error handling)
                try:
                    active_orders = await self.get_active_orders()
                    for order in active_orders:
                        order_price_int = int(order.get('price', 0))
                        order_is_ask = order.get('is_ask', False)

                        # Convert to float for comparison
                        if order_price_int > 0:
                            order_price = order_price_int / (10 ** self.price_decimals)
                            if (abs(order_price - price) < price_threshold and
                                order_is_ask == is_ask):
                                print(f"      ⚠️  Order already on exchange at ${price:.2f}, skipping")
                                return True  # Return success since order exists
                except Exception as api_err:
                    # Don't fail refill just because API check failed
                    if attempt == 0:  # Only log on first attempt
                        print(f"      ⚠️  API check failed (will still place order): {str(api_err)[:40]}")

                price_int = self.price_to_int(price)

                tx, tx_hash, err = await self.client.create_order(
                    market_index=self.market_index,
                    client_order_index=self.order_index,
                    base_amount=base_amount,
                    price=price_int,
                    is_ask=is_ask,
                    order_type=lighter.SignerClient.ORDER_TYPE_LIMIT,
                    time_in_force=lighter.SignerClient.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
                    reduce_only=False,
                    trigger_price=0
                )

                if not err:
                    # Track new order
                    self.grid_orders[self.order_index] = {
                        'price': price,
                        'is_ask': is_ask,
                        'base_amount': base_amount,
                        'price_int': price_int
                    }
                    print(f"      ✅ {'SELL' if is_ask else 'BUY'} order placed")
                    self.order_index += 1
                    return True  # Success
                else:
                    print(f"      ⚠️  Refill failed (attempt {attempt+1}/{max_retries}): {str(err)[:60]}")
                    self.order_index += 1

                    # Retry on next iteration
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                        continue
                    else:
                        return False  # Failed after all retries

            except Exception as e:
                print(f"      ❌ Refill error (attempt {attempt+1}/{max_retries}): {str(e)[:60]}")
                self.order_index += 1

                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                else:
                    return False  # Failed after all retries

        return False

    async def cancel_all_orders(self):
        """Cancel all active orders - simplified version"""
        try:
            print(f"\n🗑️  Canceling all active orders...")

            # Try multiple passes to ensure all orders are cancelled
            max_passes = 5
            total_cancelled = 0

            for pass_num in range(1, max_passes + 1):
                auth_token, err = self.client.create_auth_token_with_expiry()
                if err:
                    print(f"   ⚠️  Auth error: {err}")
                    return False

                # Fetch ALL orders at once (no pagination - simpler and safer)
                url = f"{self.base_url}/api/v1/accountActiveOrders?account_index={self.account_index}&market_id={self.market_index}&limit=500"
                headers = {"Authorization": auth_token}
                response = requests.get(url, headers=headers)
                data = response.json()

                if 'orders' not in data or len(data['orders']) == 0:
                    if pass_num == 1:
                        print(f"   ℹ️  No active orders to cancel")
                    else:
                        print(f"   ✅ All orders cancelled! (Total: {total_cancelled})")
                    return True

                orders = data['orders']

                # Use set to deduplicate by order_index (in case API returns duplicates)
                unique_orders = {}
                for order in orders:
                    order_idx = int(order.get('order_index', 0))
                    unique_orders[order_idx] = order

                orders = list(unique_orders.values())

                if pass_num == 1:
                    print(f"   📊 Total active orders found: {len(orders)}")
                else:
                    print(f"   Pass {pass_num}: {len(orders)} orders still remaining")

                success_count = 0
                failed_count = 0

                # Cancel all orders
                for i, order in enumerate(orders, 1):
                    try:
                        order_index = int(order.get('order_index', 0))
                        client_order_index = order.get('client_order_index', 'N/A')

                        tx, tx_hash, err = await self.client.cancel_order(
                            market_index=self.market_index,
                            order_index=order_index
                        )
                        if not err:
                            success_count += 1
                            if i % 20 == 0:  # Progress indicator every 20 orders
                                print(f"   Progress: {i}/{len(orders)} cancelled...")
                        else:
                            failed_count += 1
                            print(f"   ⚠️  Failed order {order_index} (COI: {client_order_index}): {str(err)[:50]}")

                        # Rate limit protection: 60 requests per 60 seconds
                        # With 80 orders, we need at least 1s per order to be safe
                        await asyncio.sleep(1.1)  # 1.1s per order

                    except Exception as e:
                        failed_count += 1
                        print(f"   ❌ Cancel error for order {i}: {str(e)[:50]}")

                total_cancelled += success_count
                print(f"   Pass {pass_num} complete: Cancelled {success_count}, Failed {failed_count}")

                # If all successful, no need for more passes
                if failed_count == 0 and len(orders) > 0:
                    print(f"   ✅ Successfully cancelled all {total_cancelled} orders!")
                    break

                # Wait before next pass
                if pass_num < max_passes:
                    print(f"   ⏳ Waiting 3 seconds before retry...")
                    await asyncio.sleep(3)

            return True

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def get_current_position(self, debug=False):
        """Get current position size using multiple methods"""
        try:
            # Method 1: Try API endpoint (may return 404)
            try:
                auth_token, err = self.client.create_auth_token_with_expiry()
                if not err:
                    url = f"{self.base_url}/api/v1/accountPositions?account_index={self.account_index}"
                    headers = {"Authorization": auth_token}
                    response = requests.get(url, headers=headers)

                    if response.status_code == 200:
                        data = response.json()

                        if debug:
                            print(f"   [DEBUG] API Response: {data}")
                            print(f"   [DEBUG] Looking for market_id: {self.market_index}")

                        if 'positions' in data:
                            for position in data['positions']:
                                market_id = int(position.get('market_id', -1))
                                base_amount = float(position.get('base_amount', 0))

                                if debug:
                                    print(f"   [DEBUG] Position: market_id={market_id}, base_amount={base_amount}")

                                if market_id == self.market_index:
                                    if debug:
                                        print(f"   [DEBUG] Match found! Returning {base_amount}")
                                    return base_amount
                        return 0
                    else:
                        if debug:
                            print(f"   [DEBUG] API returned {response.status_code}, trying alternative method...")
            except Exception as e:
                if debug:
                    print(f"   [DEBUG] API method failed: {e}, trying alternative...")

            # Method 2: Use orderbook to infer position from net fills
            # This is a fallback when API doesn't work
            if debug:
                print(f"   [DEBUG] Using alternative method: checking active orders...")

            # Get all active orders to estimate position
            try:
                auth_token, err = self.client.create_auth_token_with_expiry()
                if err:
                    return 0

                url = f"{self.base_url}/api/v1/accountActiveOrders?account_index={self.account_index}&market_id={self.market_index}&limit=100"
                headers = {"Authorization": auth_token}
                response = requests.get(url, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    if debug:
                        print(f"   [DEBUG] Active orders found: {len(data.get('orders', []))}")
                        print(f"   [DEBUG] Cannot determine exact position from orders alone")
                        print(f"   [DEBUG] Returning 0 (unable to fetch position via API)")
            except:
                pass

            return 0

        except Exception as e:
            if debug:
                print(f"   [DEBUG] Exception: {e}")
                import traceback
                traceback.print_exc()
            return 0

    async def close_all_positions(self):
        """Close all open positions with verification"""
        try:
            print(f"\n💼 Closing all positions...")
            print(f"   Current market: {self.market_symbol} (market_id={self.market_index})")

            # Enable debug mode to see what's happening
            position_size = await self.get_current_position(debug=True)

            if position_size == 0:
                print(f"   ℹ️  No open positions detected for this market")
                print(f"   If you see positions on the exchange, check:")
                print(f"     - Market ID in .env matches the position's market")
                print(f"     - Account index is correct")
                return True

            print(f"   Position size (raw from API): {position_size}")

            # Try to close position (with retries)
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                is_ask = (position_size > 0)
                action = "SELL" if is_ask else "BUY"

                # Get current market price
                current_price, best_bid, best_ask = await self.get_current_price()

                # IMPORTANT: Check if position_size is already in integer format
                # If position_size > 1000, it's likely already scaled
                if abs(position_size) > 1000:
                    # Already in integer format, use directly
                    base_amount = int(abs(position_size))
                    print(f"   Using integer format: {base_amount}")
                else:
                    # Need to scale up
                    base_amount = int(abs(position_size) * (10 ** self.size_decimals))
                    print(f"   Scaled to integer: {base_amount} (size_decimals={self.size_decimals})")

                print(f"   Attempt {attempt}/{max_retries}: Placing MARKET {action} to close position")
                print(f"   Base amount: {base_amount}")
                print(f"   Current market: bid=${best_bid}, ask=${best_ask}")

                # Use MARKET order for guaranteed execution
                # Market orders don't need a price, but we pass 0
                tx, tx_hash, err = await self.client.create_order(
                    market_index=self.market_index,
                    client_order_index=self.order_index,
                    base_amount=base_amount,
                    price=0,  # Market orders use price=0
                    is_ask=is_ask,
                    order_type=lighter.SignerClient.ORDER_TYPE_MARKET,
                    time_in_force=lighter.SignerClient.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
                    reduce_only=True,
                    trigger_price=0
                )

                self.order_index += 1

                if err:
                    print(f"   ❌ Failed to place close order!")
                    print(f"   Error type: {type(err)}")
                    print(f"   Error message: {err}")

                    # Try alternative method: aggressive limit order instead of market
                    if attempt < max_retries:
                        print(f"\n   🔄 Trying alternative method with LIMIT order...")

                        # Very aggressive pricing (20% slippage)
                        if is_ask:
                            close_price = float(best_bid) * 0.80  # 20% below best bid
                        else:
                            close_price = float(best_ask) * 1.20  # 20% above best ask

                        price_int = self.price_to_int(close_price)

                        print(f"   Placing LIMIT {action} @ ${close_price:.4f} (20% slippage)")

                        tx2, tx_hash2, err2 = await self.client.create_order(
                            market_index=self.market_index,
                            client_order_index=self.order_index,
                            base_amount=base_amount,
                            price=price_int,
                            is_ask=is_ask,
                            order_type=lighter.SignerClient.ORDER_TYPE_LIMIT,
                            time_in_force=lighter.SignerClient.ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL,
                            reduce_only=True,
                            trigger_price=0
                        )

                        self.order_index += 1

                        if err2:
                            print(f"   ❌ LIMIT order also failed: {err2}")
                        else:
                            print(f"   ✅ LIMIT close order placed")

                        await asyncio.sleep(3)
                        continue
                else:
                    print(f"   ✅ Close order placed successfully")
                    await asyncio.sleep(3)

                    # Verify position is closed
                    new_position = await self.get_current_position()
                    print(f"   Checking position... Current: {new_position}, Original: {position_size}")

                    if abs(new_position) < abs(position_size) * 0.1:  # 90% closed
                        print(f"   ✅ Position successfully closed!")
                        return True
                    else:
                        print(f"   ⚠️  Position still open after order: {new_position}")
                        print(f"   This usually means the order didn't fill or filled partially")
                        position_size = new_position
                        if attempt < max_retries:
                            await asyncio.sleep(2)
                            continue

            print(f"   ⚠️  Could not fully close position after {max_retries} attempts")
            print(f"   Final position size: {position_size}")
            print(f"   You may need to close this manually on the exchange")
            return False

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def cleanup_on_exit(self):
        """Cleanup when bot stops"""
        print(f"\n🧹 Cleaning up...")

        # Cancel all orders (with retry logic and pagination)
        await self.cancel_all_orders()

        print(f"\n📊 Final Stats:")
        print(f"   Total Trades: {self.trades_count}")
        print(f"   Completed Cycles: {self.completed_cycles:.1f}")
        print(f"   Total Volume: ${self.total_volume:.2f}")
        print(f"   Estimated Profit: ${self.total_profit:.2f}")
        print(f"   Avg Profit/Cycle: ${self.total_profit/max(self.completed_cycles, 1):.2f}")

        print(f"\n💡 REMINDER:")
        print(f"   Please close any open positions manually at:")
        print(f"   https://app.lighter.xyz")
        print(f"   Market: {self.market_symbol} (ID: {self.market_index})")

    def stop_bot(self, signum=None, frame=None):
        """Stop bot gracefully"""
        print(f"\n\n⏹️  Stopping bot...")
        self.running = False

    async def run(self):
        """Main bot execution"""
        signal.signal(signal.SIGINT, self.stop_bot)

        try:
            print("=" * 60)
            print(f"🤖 Grid Trading Bot")
            print("=" * 60)
            print(f"Market: {self.market_symbol} | Direction: {self.direction}")
            print(f"Leverage: {self.leverage}x | Grids: {self.grid_count}")
            print(f"Strategy: BUY LOW → SELL HIGH ✅")

            await self.init()
            await self.get_market_decimals()

            grid_levels, current_price = await self.calculate_grid_levels()
            await self.place_grid_orders(grid_levels, current_price)

            print(f"\n{'='*60}")
            print("✅ Grid Bot Setup Complete!")
            print(f"🔍 View orders: https://app.lighter.xyz")
            print("=" * 60)

            await self.monitor_and_refill()

        except KeyboardInterrupt:
            self.stop_bot()
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.client:
                await self.cleanup_on_exit()
                await self.client.close()
            print("\n👋 Bot stopped successfully")

if __name__ == "__main__":
    bot = GridTradingBot()
    asyncio.run(bot.run())
