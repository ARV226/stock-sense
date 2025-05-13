def get_stock_symbol(query):
    """
    Format the search query into a proper Yahoo Finance symbol for Indian stocks
    
    Args:
        query: User input query
        
    Returns:
        str: Properly formatted stock symbol
    """
    # Remove any whitespace
    query = query.strip().upper()
    
    # Common Indian stock exchanges
    indian_exchanges = ['.NS', '.BO', '.BSE', '.NSE']
    
    # Check if the query already has an exchange suffix
    has_exchange_suffix = any(query.endswith(exchange) for exchange in indian_exchanges)
    
    # If no suffix, add the NSE suffix by default (most common for Indian stocks)
    if not has_exchange_suffix:
        # If it has spaces, it might be a company name instead of a symbol
        if ' ' in query:
            # Try to extract a symbol from the company name (this is a simplified approach)
            # In a real app, you'd use a proper API to search for the correct symbol
            query = query.split(' ')[0]
        
        # Add NSE suffix
        query = f"{query}.NS"
    
    return query

def format_price(price):
    """
    Format price values for display
    
    Args:
        price: Numerical price value
        
    Returns:
        str: Formatted price string
    """
    if price >= 1000:
        return f"{price:,.2f}"
    else:
        return f"{price:.2f}"
