from typing import Dict

def cut_rod(profit_per_piece: Dict[int, int], prices: Dict[int, int], rod_length: int):
    max_profit = prices[rod_length] if rod_length in prices else -1

    for piece_size in prices.keys():
        if piece_size < rod_length:
            remaining_size = rod_length - piece_size
            if remaining_size in profit_per_piece:
                remaining_profit  = profit_per_piece[remaining_size]
            else:
                remaining_profit = cut_rod(profit_per_piece, prices, remaining_size)
                profit_per_piece[remaining_size] = remaining_profit
            total_profit = profit_per_piece[piece_size] + remaining_profit
            if total_profit > max_profit:
                max_profit = total_profit

    return max_profit 

if __name__ == '__main__':
    prices = {
        1: 1,
        2: 5,
        3: 8,
        4: 9,
        5: 10,
        6: 17,
        7: 17,
        8: 20
    }
    max_profit = {
        1: 1
    }
    profit = cut_rod(max_profit, prices, 7)
    print(profit)
