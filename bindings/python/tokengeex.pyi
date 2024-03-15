from typing import List, Optional

class TokenGeeXError(Exception):
    """
    Base class for exceptions raised by TokenGeeX.
    """

    pass

class Tokenizer:
    def encode(self, text: str) -> List[int]:
        """
        Encodes a string into a list of tokens.

        Args:
            text: The input string.

        Returns:
            A list of token IDs.

        Exceptions:
            TokenGeeXError: If the input string cannot be tokenized.
        """
        pass

    def decode(self, ids: List[int]) -> str:
        """
        Decodes a list of ids into a string.

        Args:
            ids: A list of token IDs.

        Returns:
            The decoded string.

        Exceptions:
            TokenGeeXError: If the input list of token IDs contains invalid IDs.
        """
        pass

    def token_to_id(self, token: str) -> Optional[int]:
        """
        Converts a token to its ID.

        Returns:
            The token ID or None if the token is not in the vocabulary.
        """
        pass

    def id_to_token(self, id: int) -> Optional[str]:
        """
        Converts an ID to its token.

        Returns:
            The token or None if the ID is not in the vocabulary.
        """
        pass

    def vocab_size(self) -> int:
        """
        Returns:
            The size of the vocabulary including special tokens.
        """
        pass

def load(filename: str) -> Tokenizer:
    """
    Loads a tokenizer from a file.

    Args:
        filename: The path to the tokenizer file.

    Returns:
        The tokenizer.

    Exceptions:
        TokenGeeXError: If the file cannot be loaded or is not a valid tokenizer.
    """
    pass
