def get_trade_statistics(trades, allow_short_selling=False):
    """
    Calculate and return statistics about trades.
    
    Args:
        trades: List of trades
        allow_short_selling: Whether short selling is enabled
        
    Returns:
        Dictionary with trade statistics
    """
    stats = {}
    
    # Total number of trades
    stats['total_trades'] = len(trades)
    
    # Count long and short trades
    stats['long_trades'] = len([t for t in trades if t[0] in ('Buy', 'Sell')])
    stats['short_trades'] = len([t for t in trades if t[0] in ('Short Sell', 'Cover Short')])
    
    return stats
