from tokengeex import Tokenizer

tokenizer = Tokenizer.from_file("./hub/vocab/v2/51k.json")

sentence = "Hello, world!"

# Encode (without special tokens)
ids = tokenizer.encode_ordinary(sentence)
print(ids)

# Decode
decoded = tokenizer.decode(ids, include_special_tokens=False)
print(decoded)

# Vocabulary
id = tokenizer.token_to_id(b"Hello")
assert id is not None
print(id)

token = tokenizer.id_to_token(id)
print(token)

vocab_size = tokenizer.vocab_size()
print(vocab_size)

# Special tokens
special_tokens = ["<s>", "</s>", "<pad>", "<unk>"]
tokenizer.add_special_tokens(special_tokens)
print(tokenizer.special_tokens())

sid = tokenizer.special_token_to_id("<s>")
assert sid is not None
print(sid)

sentence = "<s>Hello, world!</s>"
ids = tokenizer.encode(sentence)
assert ids[0] == sid and ids[-1] == tokenizer.special_token_to_id("</s>")
print(ids)

ids = tokenizer.encode_ordinary(sentence)
assert ids[0] != sid and ids[-1] != tokenizer.special_token_to_id("</s>")
print(ids)

new_vocab_size = tokenizer.vocab_size()
assert vocab_size == new_vocab_size - len(special_tokens)
print(vocab_size)

stoken = tokenizer.id_to_special_token(sid)
print(stoken)

# Batch encoding/decoding
sentences = ["<s>Hello, world!</s>", "<s>Hello, tokengeex!</s>"]
ids = tokenizer.encode_batch(sentences)
print(ids)

decoded = tokenizer.decode_batch(ids, include_special_tokens=True)
assert decoded == sentences
print(decoded)

decoded = tokenizer.decode_batch(ids, include_special_tokens=False)
assert decoded == ["Hello, world!", "Hello, tokengeex!"]
print(decoded)

ids = tokenizer.encode_ordinary_batch(sentences)
print(ids)

# Common prefix search
suffixes = tokenizer.common_prefix_search("self.dropout")
print([tokenizer.id_to_token(id) for id in suffixes])

# Save
tokenizer.save("/dev/null")
