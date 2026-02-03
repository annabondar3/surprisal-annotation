from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import re

from .models import get_model


def find_word_positions(text: str, words: List[str]) -> List[Tuple[int, int, str]]:
    """
    Find character positions of words in text.
    
    Args:
        text: Input text
        words: List of words to find (in order of appearance)
        
    Returns:
        List of (start_char, end_char, word) tuples
    """
    word_spans = []
    last_end = 0
    
    for word in words:
        start = text.find(word, last_end)
        if start == -1:
            raise ValueError(f"Word '{word}' not found in text after position {last_end}")
        end = start + len(word)
        word_spans.append((start, end, word))
        last_end = end
    
    return word_spans


def get_tokens_for_span(offset_mapping: List[Tuple[int, int]], 
                       start_char: int, 
                       end_char: int) -> List[int]:
    """
    Get token indices that overlap with a character span.
    
    Args:
        offset_mapping: List of (start, end) character positions for each token
        start_char: Start character position of span
        end_char: End character position of span
        
    Returns:
        List of token indices that overlap with the span
    """
    token_indices = []
    for i, (token_start, token_end) in enumerate(offset_mapping):
        if token_start < end_char and token_end > start_char:
            token_indices.append(i)
    
    return token_indices


def get_surprisal_for_words(text: str, 
                            words: List[str], 
                            model_name: str,
                            use_bos_symbol: bool = True) -> pd.DataFrame:
    """
    Calculate surprisal for specific user-defined words in text.
    
    This function provides a direct way to calculate surprisal for words
    without relying on tokenizer-specific word boundary symbols. It:
    1. Finds each word's position in the text
    2. Maps character positions to subword tokens
    3. Sums raw surprisals for tokens comprising each word
    
    Args:
        text: Input text containing the words
        words: List of words to calculate surprisal for (in order of appearance)
        model_name: Model identifier (e.g., 'pythia-70m', 'gpt2')
        use_bos_symbol: Whether to prepend BOS token (default: True)
        
    Returns:
        DataFrame with columns: word, surprisal, start_char, end_char, token_count
        
    Example:
        >>> df = get_surprisal_for_words(
        ...     text='Hello world! How are you?',
        ...     words=['Hello', 'world', 'How', 'are', 'you'],
        ...     model_name='pythia-70m'
        ... )
        >>> print(df)
    """
    model = get_model(model_name)
    
    results, offset_mapping = model.get_models_output(text, use_bos_symbol=use_bos_symbol)
    surprisals = results['surprisal']
    subwords = results['subword']
    
    word_positions = find_word_positions(text, words)
    
    word_data = []
    for start_char, end_char, word in word_positions:
        token_indices = get_tokens_for_span(offset_mapping, start_char, end_char)
        
        if not token_indices:
            raise ValueError(f"No tokens found for word '{word}' at position {start_char}-{end_char}")
        
        word_surprisal = float(np.sum(surprisals[token_indices]))
        word_tokens = [subwords[i] for i in token_indices]
        
        word_data.append({
            'word': word,
            'surprisal': word_surprisal,
            'start_char': start_char,
            'end_char': end_char,
            'token_count': len(token_indices),
            'tokens': '|'.join(word_tokens)
        })
    
    return pd.DataFrame(word_data)


def get_surprisal_for_all_words(text: str,
                                model_name: str,
                                use_bos_symbol: bool = True,
                                word_separator: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate surprisal for all words in text using simple tokenization.
    
    Args:
        text: Input text
        model_name: Model identifier
        use_bos_symbol: Whether to prepend BOS token (default: True)
        word_separator: Character(s) to split on (default: whitespace and punctuation)
        
    Returns:
        DataFrame with surprisal for each word
    """
    
    if word_separator is None:
        words = re.findall(r'\b\w+\b', text)
    else:
        words = [w for w in text.split(word_separator) if w.strip()]
    
    return get_surprisal_for_words(text, words, model_name, use_bos_symbol)
