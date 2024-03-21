import tokengeex

tokenizer = tokengeex.load("./hub/vocab/131k.json")

s = "Hello, world!"
ids = tokenizer.encode(s)

print(f"encode({s}) ->", ids)
print(f"decode({ids}) -> ", tokenizer.decode(ids))

s = "Hello"
id = tokenizer.token_to_id("Hello")
print(f"token_to_id({s}):", id)
assert id is not None
print(f"id_to_token({id}):", tokenizer.id_to_token(id))

ids = tokenizer.encode_batch(["Hello world", "Goodbye world"])
print("encode_batch(['Hello world', 'Goodbye world']) ->", ids)
print(f"decode_batch({ids}) ->", tokenizer.decode_batch(ids))

print("vocab_size() ->", tokenizer.vocab_size())

special = ["<|CODE_MIDDLE|>", "<|CODE_SUFFIX|>", "<|CODE_PREFIX|>"]
print(f"add_special_tokens({special}) ->", tokenizer.add_special_tokens(special))
print("special_tokens() ->", tokenizer.special_tokens())
print("vocab_size() ->", tokenizer.vocab_size())

sid = tokenizer.token_to_id(special[0])
assert sid is not None
print(f"token_to_id({special[0]}) ->", sid)
print(f"id_to_token({sid}) ->", tokenizer.id_to_token(sid))

print("to_string() ->", len(tokenizer.to_string()))
print("save('/dev/null') ->", tokenizer.save("/dev/null"))
