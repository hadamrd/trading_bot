# Mathematical Proof of Fee Adjustment Formula

Let's break this down step-by-step to prove why we multiply by `(1 - fee_rate) / (1 + fee_rate)` to account for fees on both buying and selling.

## Given:
- Initial investment: I
- Buy price: P_buy
- Sell price: P_sell
- Fee rate: f (assumed same for buy and sell)

## Step 1: Buying

When you buy, you pay `(1 + f)` times the price. So, the amount of asset you receive is:

```
Amount of asset = I / (P_buy * (1 + f))
```

## Step 2: Selling

When you sell, you receive `(1 - f)` times the sell price for each unit of the asset. So, your final return is:

```
Final return = (Amount of asset) * P_sell * (1 - f)
```

## Step 3: Combining Steps 1 and 2

Substituting the amount of asset from Step 1 into the equation from Step 2:

```
Final return = (I / (P_buy * (1 + f))) * P_sell * (1 - f)
```

## Step 4: Simplifying

Let's rearrange this equation:

```
Final return = I * (P_sell / P_buy) * ((1 - f) / (1 + f))
```

## Step 5: Calculating Return Rate

To get the return rate, we divide by the initial investment and subtract 1:

```
Return rate = (Final return / I) - 1
            = (P_sell / P_buy) * ((1 - f) / (1 + f)) - 1
```

This is exactly the formula used in the `get_sell_return` method!

## Proof of Correctness

To prove this is correct, let's consider what happens in the absence of fees:

1. Without fees, f = 0
2. Substituting f = 0 into our formula:
   ```
   ((1 - 0) / (1 + 0)) = 1
   ```
3. So without fees, our formula simplifies to:
   ```
   (P_sell / P_buy) - 1
   ```
   Which is the basic return calculation we'd expect.

## Conclusion

This proves that multiplying by `(1 - fee_rate) / (1 + fee_rate)` correctly accounts for the compound effect of paying fees on both the buy and sell transactions. It reduces the return by the appropriate amount to reflect these costs.