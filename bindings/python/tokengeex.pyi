from typing import Iterable, List, Optional, Tuple

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

    def encode_ordinary(self, text: str) -> List[int]:
        """
        Encodes a string into a list of tokens ignoring special tokens.

        Args:
            text: The input string.

        Returns:
            A list of token IDs.

        Exceptions:
            TokenGeeXError: If the input string cannot be tokenized.
        """
        pass

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encodes a list of strings into a list of lists of tokens.

        Args:
            texts: A list of input strings.

        Returns:
            A list of lists of token IDs.

        Exceptions:
            TokenGeeXError: If the input list of strings contains an untokenizable string.
        """
        pass

    def encode_ordinary_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encodes a list of strings into a list of lists of tokens ignoring special tokens.

        Args:
            texts: A list of input strings.

        Returns:
            A list of lists of token IDs.

        Exceptions:
            TokenGeeXError: If the input list of strings contains an untokenizable string.
        """
        pass

    def decode(self, ids: List[int], include_special_tokens: bool) -> str:
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

    def decode_batch(
        self, ids: List[List[int]], include_special_tokens: bool
    ) -> List[str]:
        """
        Decodes a list of lists of ids into a list of strings.

        Args:
            idss: A list of lists of token IDs.

        Returns:
            A list of decoded strings.

        Exceptions:
            TokenGeeXError: If the input list of lists of token IDs contains invalid IDs.
        """
        pass

    def token_to_id(self, token: bytes) -> Optional[int]:
        """
        Converts a token to its ID.

        Returns:
            The token ID or None if the token is not in the vocabulary.
        """
        pass

    def special_token_to_id(self, token: str) -> Optional[int]:
        """
        Converts a special token to its ID.

        Returns:
            The token ID or None if the token is not a special token.
        """
        pass

    def id_to_token(self, id: int) -> Optional[Tuple[bytes, float]]:
        """
        Converts an ID to its token.

        Returns:
            The token or None if the ID is not in the vocabulary.
        """
        pass

    def id_to_special_token(self, id: int) -> Optional[str]:
        """
        Converts an ID to its special token.

        Returns:
            The special token or None if the ID is not a special token.
        """
        pass

    def add_special_tokens(self, tokens: List[str]) -> None:
        """
        Adds special tokens to the tokenizer.

        Args:
            tokens: A list of special tokens.
        """
        pass

    def special_tokens(self) -> List[str]:
        """
        Returns:
            A list of special tokens.
        """
        pass

    def is_special(self, id: int) -> bool:
        """
        Returns:
            True if the token ID is a special token. False otherwise or if the
            token ID is out of bounds.
        """
        pass

    def vocab_size(self) -> int:
        """
        Returns:
            The size of the vocabulary including special tokens.
        """
        pass

    def base_vocab_size(self) -> int:
        """
        Returns:
            The size of the vocabulary including special tokens.
        """
        pass

    def special_vocab_size(self) -> int:
        """
        Returns:
            The number of special tokens in the vocabulary.
        """
        pass

    def save(self, filepath: str) -> None:
        """
        Saves the tokenizer to a file.

        Args:
            filepath: The path to the tokenizer file.
        """
        pass

    def common_prefix_search(self, text: str) -> Iterable[int]:
        """
        Iterates over all tokens that are prefixes of `text`.

        Args:
            text: The input string.

        Returns:
            A list of token IDs.
        """
        pass

    @staticmethod
    def from_file(filepath: str) -> Tokenizer:
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

    @staticmethod
    def from_str(s: str) -> Tokenizer:
        """
        Loads a tokenizer from a string.

        Args:
            s: The tokenizer string.

        Returns:
            The tokenizer.

        Exceptions:
            TokenGeeXError: If the string is not a valid tokenizer.
        """
        pass
