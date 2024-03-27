const tokengeex = require(".");
const fs = require("fs");

function readFileSyncContents(path) {
  try {
    const data = fs.readFileSync(path, { encoding: "utf8" });
    return data;
  } catch (error) {
    console.error("Error reading file:", error);
  }
}

const serialized = readFileSyncContents("../../hub/vocab/capcode-65k.json");

const state = tokengeex.fromString(serialized);
const text = "Hello, world!";
const ids = tokengeex.encode(state, text);

console.log(ids);

const decoded = tokengeex.decode(state, ids, true);

console.log(decoded);

for (let i = 0; i < ids.length; i++) {
  let { token, score } = tokengeex.idToToken(state, ids[i]);

  console.log(token.toString(), score, tokengeex.tokenToId(state, token));
}
